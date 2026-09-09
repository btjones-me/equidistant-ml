import assert from "node:assert/strict";
import test from "node:test";
import worker, {
  extractResponseOutputText,
  validateTravelTimeComparisonInput,
  validateVenueRecommendationInput
} from "../worker/index.js";

import { createDb, testCookie, testEnvironment } from "./db.mjs";
const env = testEnvironment();

function createVenueDb() {
  const db = createDb();
  db.sqlite.exec(`CREATE TABLE venue_recommendation_usage (scope TEXT, period_key TEXT, request_count INTEGER, updated_at INTEGER, PRIMARY KEY(scope, period_key))`);
  return { db, usage: {
    set(key, value) { const [scope, period] = key.split("|"); db.sqlite.prepare("INSERT OR REPLACE INTO venue_recommendation_usage VALUES (?, ?, ?, ?)").run(scope, period, value.request_count, value.updated_at); },
    values() { return db.sqlite.prepare("SELECT * FROM venue_recommendation_usage").all(); }
  } };
}
function createTravelTimeDb() { return { db: createDb() }; }

function comparisonRequestBody(friendCount = 1) {
  const friends = Array.from({ length: friendCount }, (_, index) => ({
    name: `Friend ${index + 1}`,
    lat: 51.52 + index / 1000,
    lng: -0.14 - index / 1000
  }));
  return {
    friends,
    included_friend_indexes: friends.map((_, index) => index),
    combine: "balanced",
    model_cells: [
      {
        destination_id: "cell-1",
        lat: 51.51,
        lng: -0.13,
        x_index: 0,
        y_index: 0,
        h3_cell: "cell-1",
        h3_resolution: 9,
        boundary: [[51.509, -0.131], [51.511, -0.131], [51.51, -0.129]],
        model_score_minutes: 10,
        ...Object.fromEntries(friends.map((_, index) => [`friend_${index}_model_minutes`, 10 + index]))
      },
      {
        destination_id: "cell-2",
        lat: 51.53,
        lng: -0.11,
        x_index: 1,
        y_index: 0,
        h3_cell: "cell-2",
        h3_resolution: 9,
        boundary: [[51.529, -0.111], [51.531, -0.111], [51.53, -0.109]],
        model_score_minutes: 20,
        ...Object.fromEntries(friends.map((_, index) => [`friend_${index}_model_minutes`, 20 + index]))
      }
    ]
  };
}

async function accessCookie() { return testCookie; }

function providerFetchMock(requests, { recommendationIds = [1, 2, 3] } = {}) {
  return async (url, init = {}) => {
    const href = String(url);
    requests.push({ href, init });
    if (href.includes("places.googleapis.com/v1/places:searchText")) {
      return Response.json({
        places: [1, 2, 3, 4].map((index) => ({
          id: `google-place-${index}`,
          displayName: { text: `Venue ${index}` },
          formattedAddress: `${index} Test Street, London`,
          location: { latitude: 51.51 + index / 1000, longitude: -0.12 - index / 1000 },
          primaryType: "pub",
          types: ["pub", "restaurant"],
          rating: 4.2 + index / 10,
          userRatingCount: 100 * index,
          priceLevel: "PRICE_LEVEL_MODERATE",
          currentOpeningHours: { openNow: true, weekdayDescriptions: ["Wednesday: 12:00-23:00"] },
          websiteUri: `https://venue-${index}.example/`,
          googleMapsUri: `https://maps.google.com/?cid=${index}`,
          photos: [{
            name: `places/google-place-${index}/photos/${index === 1 ? "p".repeat(600) : `photo-${index}`}`,
            authorAttributions: [{ displayName: "Test photographer", uri: "https://example.com/photographer" }]
          }]
        }))
      });
    }
    if (href === "https://api.openai.com/v1/responses") {
      return Response.json({
        output: [
          {
            type: "web_search_call",
            action: { sources: [1, 2, 3].map((index) => ({ url: `https://venue-${index}.example/details` })) }
          },
          {
            type: "message",
            content: [{
              type: "output_text",
              text: JSON.stringify({
                recommendations: recommendationIds.map((index) => ({
                  place_id: `google-place-${index}`,
                  why: `Venue ${index} is a strong group match.`,
                  verified_details: ["Open this evening", "Accepts groups"],
                  source_urls: [`https://venue-${index}.example/details`]
                }))
              })
            }]
          }
        ]
      });
    }
    throw new Error(`Unexpected provider request: ${href}`);
  };
}

test("authenticated public paths map to the private asset namespace", async () => {
  let requestedPath = "";
  const mappedEnv = {
    ...env,
    ASSETS: {
      fetch: async (request) => {
        requestedPath = new URL(request.url).pathname;
        return new Response("model bytes");
      }
    }
  };
  const cookie = await accessCookie();

  await worker.fetch(
    new Request("https://example.test/model/model.u8", { headers: { Origin: "https://example.test", Cookie: cookie } }),
    mappedEnv
  );

  assert.equal(
    requestedPath,
    "/__EQUIDISTANT_ASSET_NAMESPACE__/model/model.u8"
  );
});

test("hosted HTML receives an absolute social preview URL", async () => {
  const cookie = await accessCookie();
  const htmlEnv = {
    ...env,
    AUTH_ORIGIN: "https://equidistant.example",
    ASSETS: {
      fetch: async () => new Response(
        '<meta property="og:image" content="__EQUIDISTANT_ORIGIN__/og.png">',
        { headers: { "Content-Type": "text/html" } }
      )
    }
  };

  const response = await worker.fetch(
    new Request("https://equidistant.example/", { headers: { Origin: "https://example.test", Cookie: cookie } }),
    htmlEnv
  );
  const html = await response.text();

  assert.match(html, /https:\/\/equidistant\.example\/og\.png/);
  assert.doesNotMatch(html, /__EQUIDISTANT_ORIGIN__/);
});

test("venue request validation constrains text and London coordinates", () => {
  assert.deepEqual(validateVenueRecommendationInput({
    query: "  relaxed   pub  ",
    area_name: "Soho",
    lat: 51.513,
    lng: -0.132
  }), {
    query: "relaxed pub",
    areaName: "Soho",
    lat: 51.513,
    lng: -0.132
  });
  assert.throws(
    () => validateVenueRecommendationInput({ query: "pub", area_name: "Paris", lat: 48.85, lng: 2.35 }),
    /London coverage/
  );
});

test("TravelTime comparison validation keeps only safe model surface fields", () => {
  const input = comparisonRequestBody();
  input.model_cells[0].untrusted = "discard me";
  const validated = validateTravelTimeComparisonInput(input);

  assert.equal(validated.friends.length, 1);
  assert.equal(validated.cells.length, 2);
  assert.equal(validated.cells[0].model_score_minutes, 10);
  assert.equal(validated.cells[0].untrusted, undefined);
  assert.throws(
    () => validateTravelTimeComparisonInput({ ...input, friends: [{ lat: 48.85, lng: 2.35 }] }),
    /within London/
  );
});

test("hosted comparisons merge TravelTime references without exposing credentials", async (context) => {
  const { db } = createTravelTimeDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const body = JSON.parse(init.body);
    requests.push({ href: String(url), init, body });
    return Response.json({
      results: body.arrival_searches.one_to_many.map((search) => ({
        search_id: search.id,
        locations: [
          { id: "d_0", properties: { travel_time: 720 } },
          { id: "d_1", properties: { travel_time: 1080 } }
        ],
        unreachable: []
      }))
    });
  };
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();
  const response = await worker.fetch(new Request("https://example.test/api/comparison-surface", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.80" },
    body: JSON.stringify(comparisonRequestBody())
  }), {
    ...env,
    DB: db,
    RATE_LIMIT_SECRET: "test-rate-secret",
    TRAVELTIME_APP_ID: "test-app-id",
    TRAVELTIME_API_KEY: "test-api-key"
  });
  const result = await response.json();

  assert.equal(response.status, 200);
  assert.equal(result.cells[0].reference_score_minutes, 12);
  assert.equal(result.cells[0].signed_error_minutes, -2);
  assert.equal(result.cells[1].reference_score_minutes, 18);
  assert.equal(result.cells[1].signed_error_minutes, 2);
  assert.equal(result.metadata.comparison.mae_minutes, 2);
  assert.equal(result.metadata.comparison.mean_signed_error_minutes, 0);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].href, "https://api.traveltimeapp.com/v4/time-filter/fast");
  assert.equal(requests[0].body.arrival_searches.one_to_many[0].arrival_time_period, "weekday_morning");
  assert.equal(requests[0].init.headers["X-Application-Id"], "test-app-id");
  assert.doesNotMatch(JSON.stringify(result), /test-(app-id|api-key)/);
});

test("TravelTime comparisons are capped by requested origins per visitor", async (context) => {
  const { db } = createTravelTimeDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init = {}) => {
    const body = JSON.parse(init.body);
    requests.push(body);
    return Response.json({
      results: body.arrival_searches.one_to_many.map((search) => ({
        search_id: search.id,
        locations: [
          { id: "d_0", properties: { travel_time: 720 } },
          { id: "d_1", properties: { travel_time: 1080 } }
        ],
        unreachable: []
      }))
    });
  };
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();
  const comparisonEnv = {
    ...env,
    DB: db,
    RATE_LIMIT_SECRET: "test-rate-secret",
    TRAVELTIME_APP_ID: "test-app-id",
    TRAVELTIME_API_KEY: "test-api-key"
  };
  const makeRequest = () => new Request("https://example.test/api/comparison-surface", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.81" },
    body: JSON.stringify(comparisonRequestBody(6))
  });

  assert.equal((await worker.fetch(makeRequest(), comparisonEnv)).status, 200);
  assert.equal((await worker.fetch(makeRequest(), comparisonEnv)).status, 200);
  const blocked = await worker.fetch(makeRequest(), comparisonEnv);
  assert.equal(blocked.status, 429);
  assert.match((await blocked.json()).detail, /hourly TravelTime comparison limit/);
  assert.equal(requests.length, 2);
});

test("Responses API output text is extracted without relying on an SDK helper", () => {
  assert.equal(extractResponseOutputText({
    output: [{ type: "message", content: [{ type: "output_text", text: "{\"ok\":true}" }] }]
  }), "{\"ok\":true}");
});

test("venue recommendations combine Google facts with researched OpenAI selections", async (context) => {
  const { db } = createVenueDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = providerFetchMock(requests);
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();
  const recommendationEnv = {
    ...env,
    DB: db,
    RATE_LIMIT_SECRET: "test-rate-secret",
    GOOGLE_PLACES_API_KEY: "test-google-key",
    OPENAI_API_KEY: "test-openai-key"
  };
  const response = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.70" },
    body: JSON.stringify({ query: "A relaxed pub", area_name: "Soho", lat: 51.513, lng: -0.132 })
  }), recommendationEnv);
  const result = await response.json();

  assert.equal(response.status, 200);
  assert.equal(result.places.length, 3);
  assert.equal(result.places[0].name, "Venue 1");
  assert.equal(result.places[0].why, "Venue 1 is a strong group match.");
  assert.match(result.places[0].photo_url, /^\/api\/place-photo\?name=/);
  assert.ok(new URL(`https://example.test${result.places[0].photo_url}`).searchParams.get("name").length > 420);
  assert.doesNotMatch(JSON.stringify(result), /test-(google|openai)-key/);

  const openAIRequest = requests.find((request) => request.href === "https://api.openai.com/v1/responses");
  const openAIBody = JSON.parse(openAIRequest.init.body);
  assert.equal(openAIBody.model, "gpt-5.6-luna");
  assert.equal(openAIBody.reasoning.effort, "medium");
  assert.equal(openAIBody.tools[0].type, "web_search");
  assert.equal(openAIBody.text.format.type, "json_schema");
  assert.deepEqual(
    openAIBody.text.format.schema.properties.recommendations.items.properties.place_id.enum,
    ["google-place-1", "google-place-2", "google-place-3", "google-place-4"]
  );
});

test("duplicate research selections are completed from verified Google candidates", async (context) => {
  const { db } = createVenueDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = providerFetchMock(requests, { recommendationIds: [1, 1, 2] });
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();

  const response = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.73" },
    body: JSON.stringify({ query: "A wine bar", area_name: "Soho", lat: 51.513, lng: -0.132 })
  }), {
    ...env,
    DB: db,
    GOOGLE_PLACES_API_KEY: "test-google-key",
    OPENAI_API_KEY: "test-openai-key"
  });
  const result = await response.json();

  assert.equal(response.status, 200);
  assert.deepEqual(result.places.map((place) => place.place_id), [
    "google-place-1",
    "google-place-2",
    "google-place-3"
  ]);
  assert.match(result.places[2].verified_details[0], /Google rating/);
  assert.equal(result.places[2].source_urls[0], "https://venue-3.example/");
});

test("identical venue searches use the D1 cache without another paid request", async (context) => {
  const { db } = createVenueDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = providerFetchMock(requests);
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();
  const recommendationEnv = {
    ...env,
    DB: db,
    GOOGLE_PLACES_API_KEY: "test-google-key",
    OPENAI_API_KEY: "test-openai-key"
  };
  const makeRequest = () => new Request("https://example.test/api/venue-recommendations", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.71" },
    body: JSON.stringify({ query: "A quiet museum", area_name: "Soho", lat: 51.513, lng: -0.132 })
  });

  const first = await worker.fetch(makeRequest(), recommendationEnv);
  const second = await worker.fetch(makeRequest(), recommendationEnv);
  const secondBody = await second.json();

  assert.equal(first.status, 200);
  assert.equal(second.status, 200);
  assert.equal(secondBody.cached, true);
  assert.equal(requests.length, 2);
});

test("nearby searches from different visitors cannot reuse another request's area", async (context) => {
  const { db } = createVenueDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = providerFetchMock(requests);
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();
  const recommendationEnv = {
    ...env,
    DB: db,
    GOOGLE_PLACES_API_KEY: "test-google-key",
    OPENAI_API_KEY: "test-openai-key"
  };
  const search = (body, visitor) => worker.fetch(new Request("https://example.test/api/venue-recommendations", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": visitor },
    body: JSON.stringify(body)
  }), recommendationEnv);
  const first = await search({ query: "A quiet museum", area_name: "First visitor's meeting area", lat: 51.5131, lng: -0.1321 }, "203.0.113.74");
  assert.equal(first.status, 200);
  const second = await search({ query: "A QUIET MUSEUM", area_name: "Second visitor's meeting area", lat: 51.5132, lng: -0.1322 }, "203.0.113.75");
  assert.equal(second.status, 200);
  const body = await second.json();
  assert.deepEqual(body.area, { name: "Second visitor's meeting area", lat: 51.5132, lng: -0.1322 });
  assert.equal(body.query, "A QUIET MUSEUM");
  assert.equal(body.cached, false);
  assert.equal(requests.length, 4);
});

test("place photos still load when the deployment edge cache is unavailable", async (context) => {
  const { db } = createVenueDb();
  const originalFetch = globalThis.fetch;
  const originalCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  const originalWarn = console.warn;
  const warnings = [];
  const photoName = `places/place-1/photos/${"p".repeat(600)}`;
  console.warn = (...values) => warnings.push(values.join(" "));
  globalThis.fetch = async (url) => {
    const href = String(url);
    if (href.startsWith(`https://places.googleapis.com/v1/${photoName}/media`)) {
      return Response.json({ photoUri: "https://images.example.com/photo.jpg" });
    }
    if (href === "https://images.example.com/photo.jpg") {
      return new Response(new Uint8Array([255, 216, 255, 217]), {
        headers: { "Content-Type": "image/jpeg" }
      });
    }
    throw new Error(`Unexpected photo request: ${href}`);
  };
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: {
      default: {
        match: async () => { throw new Error("edge cache unavailable"); },
        put: async () => {}
      }
    }
  });
  context.after(() => {
    globalThis.fetch = originalFetch;
    console.warn = originalWarn;
    if (originalCaches) {
      Object.defineProperty(globalThis, "caches", originalCaches);
    } else {
      delete globalThis.caches;
    }
  });
  const cookie = await accessCookie();

  const response = await worker.fetch(new Request(
    `https://example.test/api/place-photo?name=${encodeURIComponent(photoName)}`,
    { headers: { Origin: "https://example.test", Cookie: cookie } }
  ), { ...env, DB: db, GOOGLE_PLACES_API_KEY: "test-google-key" });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "image/jpeg");
  assert.equal((await response.arrayBuffer()).byteLength, 4);
  assert.ok(warnings.some((message) => message.includes("Place photo edge cache unavailable")));
});

test("place photo cache keys ignore unrelated query parameters", async (context) => {
  const { db, usage } = createVenueDb();
  const originalFetch = globalThis.fetch;
  const originalCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  const cache = new Map();
  const requests = [];
  const photoName = "places/place-1/photos/photo-1";
  globalThis.fetch = async (url) => {
    const href = String(url);
    requests.push(href);
    if (href.startsWith(`https://places.googleapis.com/v1/${photoName}/media`)) {
      return Response.json({ photoUri: "https://images.example.com/photo.jpg" });
    }
    if (href === "https://images.example.com/photo.jpg") {
      return new Response(new Uint8Array([255, 216, 255, 217]), {
        headers: { "Content-Type": "image/jpeg" }
      });
    }
    throw new Error(`Unexpected photo request: ${href}`);
  };
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: {
      default: {
        match: async (request) => cache.get(request.url)?.clone() ?? null,
        put: async (request, response) => cache.set(request.url, response.clone())
      }
    }
  });
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalCaches) {
      Object.defineProperty(globalThis, "caches", originalCaches);
    } else {
      delete globalThis.caches;
    }
  });
  const cookie = await accessCookie();
  const photoEnv = { ...env, DB: db, GOOGLE_PLACES_API_KEY: "test-google-key" };
  const makeRequest = (nonce) => new Request(
    `https://example.test/api/place-photo?name=${encodeURIComponent(photoName)}&nonce=${nonce}`,
    { headers: { Origin: "https://example.test", Cookie: cookie } }
  );

  const first = await worker.fetch(makeRequest("first"), photoEnv);
  const second = await worker.fetch(makeRequest("second"), photoEnv);

  assert.equal(first.status, 200);
  assert.equal(second.status, 200);
  assert.equal(requests.length, 2);
  assert.deepEqual([...cache.keys()], [
    `https://example.test/api/place-photo?name=${encodeURIComponent(photoName)}`
  ]);
  assert.equal([...usage.values()].every((row) => row.request_count === 1), true);
});

test("place photos fail closed before Google when cost controls are unavailable", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  let fetched = false;
  globalThis.fetch = async () => {
    fetched = true;
    throw new Error("Google should not be called");
  };
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: { default: { match: async () => null, put: async () => {} } }
  });
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalCaches) {
      Object.defineProperty(globalThis, "caches", originalCaches);
    } else {
      delete globalThis.caches;
    }
  });
  const cookie = await accessCookie();

  const response = await worker.fetch(new Request(
    "https://example.test/api/place-photo?name=places%2Fplace-1%2Fphotos%2Fphoto-1",
    { headers: { Origin: "https://example.test", Cookie: cookie } }
  ), { ...env, DB: undefined, GOOGLE_PLACES_API_KEY: "test-google-key" });

  assert.equal(response.status, 503);
  assert.match(await response.text(), /temporarily unavailable/);
  assert.equal(fetched, false);
});

test("place photos stop before Google at the monthly free-tier buffer", async (context) => {
  const { db, usage } = createVenueDb();
  const originalFetch = globalThis.fetch;
  const originalCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  let fetched = false;
  const now = Math.floor(Date.now() / 1000);
  const month = new Date(now * 1000).toISOString().slice(0, 7);
  usage.set(`global:photo:month|${month}`, { request_count: 900, updated_at: now });
  globalThis.fetch = async () => {
    fetched = true;
    throw new Error("Google should not be called");
  };
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: { default: { match: async () => null, put: async () => {} } }
  });
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalCaches) {
      Object.defineProperty(globalThis, "caches", originalCaches);
    } else {
      delete globalThis.caches;
    }
  });
  const cookie = await accessCookie();

  const response = await worker.fetch(new Request(
    "https://example.test/api/place-photo?name=places%2Fplace-1%2Fphotos%2Fphoto-1",
    { headers: { Origin: "https://example.test", Cookie: cookie } }
  ), { ...env, DB: db, GOOGLE_PLACES_API_KEY: "test-google-key" });

  assert.equal(response.status, 429);
  assert.match((await response.json()).detail, /month's live photo allowance/);
  assert.equal(response.headers.get("Retry-After"), "86400");
  assert.equal(fetched, false);
});

test("uncached venue research is limited to five searches per account per rolling hour", async (context) => {
  const { db } = createVenueDb();
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = providerFetchMock(requests);
  context.after(() => { globalThis.fetch = originalFetch; });
  const cookie = await accessCookie();
  const recommendationEnv = {
    ...env,
    DB: db,
    ADMIN_EMAILS: "",
    GOOGLE_PLACES_API_KEY: "test-google-key",
    OPENAI_API_KEY: "test-openai-key"
  };

  for (let index = 0; index < 5; index += 1) {
    const response = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
      method: "POST",
      headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.72" },
      body: JSON.stringify({ query: `Group dinner option ${index}`, area_name: "Soho", lat: 51.513, lng: -0.132 })
    }), recommendationEnv);
    assert.equal(response.status, 200);
  }

  const blocked = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
    method: "POST",
    headers: { Origin: "https://example.test", Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.72" },
    body: JSON.stringify({ query: "A sixth uncached search", area_name: "Soho", lat: 51.513, lng: -0.132 })
  }), recommendationEnv);
  assert.equal(blocked.status, 429);
  assert.match((await blocked.json()).detail, /last 60 minutes/);
  assert.equal(requests.length, 10);
});

test("recommendation caches never reuse another Google account's query result", async (context) => {
  const { createSession } = await import("../worker/auth.js");
  const { db } = createVenueDb();
  const otherCookie = (await createSession(db, { id: "another-google-account", email: "other@example.test", name: "Other" })).split(";")[0];
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = providerFetchMock(requests);
  context.after(() => { globalThis.fetch = originalFetch; });
  const recommendationEnv = { ...env, DB: db, GOOGLE_PLACES_API_KEY: "test-google-key", OPENAI_API_KEY: "test-openai-key" };
  for (const cookie of [testCookie, otherCookie]) {
    const response = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
      method: "POST", headers: { Cookie: cookie, Origin: "https://example.test", "Content-Type": "application/json" },
      body: JSON.stringify({ query: "Dinner for a private group", area_name: "Soho", lat: 51.513, lng: -0.132 })
    }), recommendationEnv);
    assert.equal(response.status, 200);
    assert.equal((await response.json()).cached, false);
  }
  assert.equal(requests.filter((request) => request.href === "https://api.openai.com/v1/responses").length, 2);
});
