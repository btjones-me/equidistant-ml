import assert from "node:assert/strict";
import test from "node:test";
import worker, {
  extractResponseOutputText,
  validateTravelTimeComparisonInput,
  validateVenueRecommendationInput
} from "../worker/index.js";

const env = {
  SITE_PASSWORD: "test-password-do-not-use",
  ASSETS: {
    fetch: async () => new Response("protected app", { headers: { "Content-Type": "text/plain" } })
  }
};

function createRateLimitDb() {
  const rows = new Map();
  const visitors = new Map();
  const db = {
    async batch(statements) {
      return Promise.all(statements.map((statement) => statement.run()));
    },
    prepare(sql) {
      let parameters = [];
      const statement = {
        bind(...values) {
          parameters = values;
          return statement;
        },
        async first() {
          if (sql.includes("SELECT window_started_at")) {
            return rows.get(parameters[0]) ?? null;
          }
          if (sql.includes("COUNT(*) AS all_time")) {
            const visitorRows = [...visitors.values()];
            return {
              all_time: visitorRows.length,
              last_24h: visitorRows.filter((row) => row.last_seen_at >= parameters[0]).length,
              last_7d: visitorRows.filter((row) => row.last_seen_at >= parameters[1]).length,
              last_30d: visitorRows.filter((row) => row.last_seen_at >= parameters[2]).length,
              total_unlocks: visitorRows.reduce((total, row) => total + row.unlocks, 0),
              last_seen_at: visitorRows.length
                ? Math.max(...visitorRows.map((row) => row.last_seen_at))
                : null
            };
          }
          throw new Error(`Unexpected query: ${sql}`);
        },
        async all() {
          if (!sql.includes("FROM analytics_visitors")) {
            throw new Error(`Unexpected query: ${sql}`);
          }
          const locations = new Map();
          for (const row of visitors.values()) {
            const locationKey = `${row.country}|${row.region}|${row.city}`;
            const current = locations.get(locationKey) ?? {
              country: row.country ?? "Unknown",
              region: row.region ?? "Unknown",
              city: row.city ?? "Unknown",
              unique_visitors: 0,
              unlocks: 0
            };
            current.unique_visitors += 1;
            current.unlocks += row.unlocks;
            locations.set(locationKey, current);
          }
          return { results: [...locations.values()] };
        },
        async run() {
          if (sql.includes("CREATE TABLE")) {
            return { success: true };
          }
          if (sql.includes("INSERT INTO unlock_rate_limits")) {
            const [clientKey, now, windowSeconds] = parameters;
            const current = rows.get(clientKey);
            if (!current || now - current.window_started_at >= windowSeconds) {
              rows.set(clientKey, {
                window_started_at: now,
                failures: 1,
                updated_at: now
              });
            } else {
              current.failures += 1;
              current.updated_at = now;
            }
            return { success: true };
          }
          if (sql.includes("INSERT INTO analytics_visitors")) {
            const [visitorKey, now, country, region, city] = parameters;
            const current = visitors.get(visitorKey);
            visitors.set(visitorKey, current
              ? {
                  ...current,
                  last_seen_at: now,
                  unlocks: current.unlocks + 1,
                  country: country ?? current.country,
                  region: region ?? current.region,
                  city: city ?? current.city
                }
              : {
                  first_seen_at: now,
                  last_seen_at: now,
                  unlocks: 1,
                  country,
                  region,
                  city
                });
            return { success: true };
          }
          if (sql.includes("updated_at <")) {
            for (const [key, row] of rows) {
              if (row.updated_at < parameters[0]) {
                rows.delete(key);
              }
            }
            return { success: true };
          }
          if (sql.includes("client_key =")) {
            rows.delete(parameters[0]);
            return { success: true };
          }
          throw new Error(`Unexpected query: ${sql}`);
        }
      };
      return statement;
    }
  };
  return { db, rows, visitors };
}

function createVenueDb() {
  const cache = new Map();
  const usage = new Map();
  return {
    cache,
    usage,
    db: {
      async batch(statements) {
        return Promise.all(statements.map((statement) => statement.run()));
      },
      prepare(sql) {
        let parameters = [];
        const statement = {
          bind(...values) {
            parameters = values;
            return statement;
          },
          async first() {
            if (sql.includes("FROM venue_recommendation_cache")) {
              const row = cache.get(parameters[0]);
              return row && row.expires_at > parameters[1] ? { payload_json: row.payload_json } : null;
            }
            if (sql.includes("INSERT INTO venue_recommendation_usage")) {
              const key = `${parameters[0]}|${parameters[1]}`;
              const requestCount = (usage.get(key)?.request_count ?? 0) + 1;
              usage.set(key, { request_count: requestCount, updated_at: parameters[2] });
              return { request_count: requestCount };
            }
            throw new Error(`Unexpected venue query: ${sql}`);
          },
          async run() {
            if (sql.includes("CREATE TABLE")) {
              return { success: true };
            }
            if (sql.includes("INSERT INTO venue_recommendation_cache")) {
              cache.set(parameters[0], {
                payload_json: parameters[1],
                created_at: parameters[2],
                expires_at: parameters[3]
              });
              return { success: true };
            }
            if (sql.includes("DELETE FROM venue_recommendation_cache WHERE cache_key")) {
              cache.delete(parameters[0]);
              return { success: true };
            }
            if (sql.includes("DELETE FROM venue_recommendation_cache")) {
              for (const [key, row] of cache) {
                if (row.expires_at <= parameters[0]) {
                  cache.delete(key);
                }
              }
              return { success: true };
            }
            if (sql.includes("DELETE FROM venue_recommendation_usage")) {
              for (const [key, row] of usage) {
                if (row.updated_at < parameters[0]) {
                  usage.delete(key);
                }
              }
              return { success: true };
            }
            throw new Error(`Unexpected venue run: ${sql}`);
          }
        };
        return statement;
      }
    }
  };
}

function createTravelTimeDb() {
  const usage = new Map();
  return {
    usage,
    db: {
      async batch(statements) {
        return Promise.all(statements.map((statement) => statement.run()));
      },
      prepare(sql) {
        let parameters = [];
        const statement = {
          bind(...values) {
            parameters = values;
            return statement;
          },
          async first() {
            if (sql.includes("INSERT INTO traveltime_comparison_usage")) {
              const key = `${parameters[0]}|${parameters[1]}`;
              const originCount = (usage.get(key)?.origin_count ?? 0) + parameters[2];
              usage.set(key, { origin_count: originCount, updated_at: parameters[3] });
              return { origin_count: originCount };
            }
            throw new Error(`Unexpected TravelTime query: ${sql}`);
          },
          async run() {
            if (sql.includes("CREATE TABLE")) {
              return { success: true };
            }
            if (sql.includes("DELETE FROM traveltime_comparison_usage")) {
              for (const [key, row] of usage) {
                if (row.updated_at < parameters[0]) {
                  usage.delete(key);
                }
              }
              return { success: true };
            }
            throw new Error(`Unexpected TravelTime run: ${sql}`);
          }
        };
        return statement;
      }
    }
  };
}

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

async function accessCookie() {
  const response = await worker.fetch(new Request("https://example.test/unlock", {
    method: "POST",
    body: new URLSearchParams({ password: env.SITE_PASSWORD })
  }), env);
  return response.headers.get("set-cookie").split(";", 1)[0];
}

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

test("password gate hides every asset until unlocked", async () => {
  const locked = await worker.fetch(new Request("https://example.test/"), env);
  assert.equal(locked.status, 200);
  assert.match(await locked.text(), /Private preview/);
  assert.equal(locked.headers.get("x-frame-options"), "DENY");
  assert.match(locked.headers.get("content-security-policy"), /https:\/\/www\.googletagmanager\.com/);
  assert.match(locked.headers.get("content-security-policy"), /https:\/\/\*\.google-analytics\.com/);
});

test("incorrect passwords are rejected", async () => {
  const body = new URLSearchParams({ password: "wrong" });
  const response = await worker.fetch(
    new Request("https://example.test/unlock", { method: "POST", body }),
    env
  );
  assert.equal(response.status, 401);
  assert.match(await response.text(), /not correct/);
});

test("password failures are limited per client IP", async () => {
  const { db } = createRateLimitDb();
  const limitedEnv = { ...env, DB: db };
  const requestFor = (ip) => new Request("https://example.test/unlock", {
    method: "POST",
    headers: { "CF-Connecting-IP": ip },
    body: new URLSearchParams({ password: "wrong" })
  });

  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await worker.fetch(requestFor("203.0.113.10"), limitedEnv);
    assert.equal(response.status, 401);
  }

  const blocked = await worker.fetch(requestFor("203.0.113.10"), limitedEnv);
  assert.equal(blocked.status, 429);
  assert.ok(Number(blocked.headers.get("retry-after")) > 0);
  assert.match(await blocked.text(), /Too many incorrect attempts/);

  const otherClient = await worker.fetch(requestFor("203.0.113.11"), limitedEnv);
  assert.equal(otherClient.status, 401);
});

test("successful unlock clears prior failures", async () => {
  const { db, rows } = createRateLimitDb();
  const limitedEnv = { ...env, DB: db };
  const headers = { "CF-Connecting-IP": "203.0.113.12" };

  await worker.fetch(
    new Request("https://example.test/unlock", {
      method: "POST",
      headers,
      body: new URLSearchParams({ password: "wrong" })
    }),
    limitedEnv
  );
  assert.equal(rows.size, 1);

  const unlocked = await worker.fetch(
    new Request("https://example.test/unlock", {
      method: "POST",
      headers,
      body: new URLSearchParams({ password: env.SITE_PASSWORD })
    }),
    limitedEnv
  );
  assert.equal(unlocked.status, 303);
  assert.equal(rows.size, 0);
});

test("usage analytics count browser visitors without storing raw IPs", async () => {
  const { db, visitors } = createRateLimitDb();
  const analyticsEnv = {
    ...env,
    DB: db,
    ANALYTICS_SECRET: "test-analytics-secret-do-not-use"
  };
  const makeUnlockRequest = (cookie = "") => {
    const request = new Request("https://example.test/unlock", {
      method: "POST",
      headers: {
        "CF-Connecting-IP": "203.0.113.40",
        ...(cookie ? { Cookie: cookie } : {})
      },
      body: new URLSearchParams({ password: env.SITE_PASSWORD })
    });
    Object.defineProperty(request, "cf", {
      value: { country: "GB", region: "England", city: "London" }
    });
    return request;
  };

  const firstUnlock = await worker.fetch(makeUnlockRequest(), analyticsEnv);
  const setCookie = firstUnlock.headers.get("set-cookie");
  const accessCookie = /equidistant_access=[^;,]+/.exec(setCookie)?.[0];
  const visitorCookie = /__Host-equidistant_visitor=[^;,]+/.exec(setCookie)?.[0];
  assert.ok(accessCookie);
  assert.ok(visitorCookie);

  const cookie = `${accessCookie}; ${visitorCookie}`;
  await worker.fetch(makeUnlockRequest(cookie), analyticsEnv);
  const usageResponse = await worker.fetch(
    new Request("https://example.test/api/usage", { headers: { Cookie: cookie } }),
    analyticsEnv
  );
  const usage = await usageResponse.json();

  assert.equal(usageResponse.status, 200);
  assert.equal(usage.unique_visitors.all_time, 1);
  assert.equal(usage.unique_visitors.last_24h, 1);
  assert.equal(usage.total_unlocks, 2);
  assert.deepEqual(usage.locations[0], {
    country: "GB",
    region: "England",
    city: "London",
    unique_visitors: 1,
    unlocks: 2
  });
  assert.doesNotMatch(JSON.stringify([...visitors]), /203\.0\.113\.40/);
});

test("a successful unlock grants access with a secure cookie", async () => {
  const body = new URLSearchParams({ password: env.SITE_PASSWORD });
  const unlocked = await worker.fetch(
    new Request("https://example.test/unlock", { method: "POST", body }),
    env
  );
  assert.equal(unlocked.status, 303);
  const setCookie = unlocked.headers.get("set-cookie");
  assert.match(setCookie, /HttpOnly/);
  assert.match(setCookie, /Secure/);
  assert.match(setCookie, /SameSite=Strict/);

  const cookie = setCookie.split(";", 1)[0];
  const appResponse = await worker.fetch(
    new Request("https://example.test/", { headers: { Cookie: cookie } }),
    env
  );
  assert.equal(appResponse.status, 200);
  assert.equal(await appResponse.text(), "protected app");
});

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
  const body = new URLSearchParams({ password: env.SITE_PASSWORD });
  const unlocked = await worker.fetch(
    new Request("https://example.test/unlock", { method: "POST", body }),
    mappedEnv
  );
  const cookie = unlocked.headers.get("set-cookie").split(";", 1)[0];

  await worker.fetch(
    new Request("https://example.test/model/model.u8", { headers: { Cookie: cookie } }),
    mappedEnv
  );

  assert.equal(
    requestedPath,
    "/__EQUIDISTANT_ASSET_NAMESPACE__/model/model.u8"
  );
});

test("hosted HTML receives an absolute social preview URL", async () => {
  const body = new URLSearchParams({ password: env.SITE_PASSWORD });
  const unlocked = await worker.fetch(
    new Request("https://equidistant.example/unlock", { method: "POST", body }),
    env
  );
  const cookie = unlocked.headers.get("set-cookie").split(";", 1)[0];
  const htmlEnv = {
    ...env,
    ASSETS: {
      fetch: async () => new Response(
        '<meta property="og:image" content="__EQUIDISTANT_ORIGIN__/og.png">',
        { headers: { "Content-Type": "text/html" } }
      )
    }
  };

  const response = await worker.fetch(
    new Request("https://equidistant.example/", { headers: { Cookie: cookie } }),
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
    headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.80" },
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
    headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.81" },
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
    headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.70" },
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
  assert.equal(openAIBody.model, "gpt-5.6-terra");
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
    headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.73" },
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
    headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.71" },
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

test("place photos still load when the deployment edge cache is unavailable", async (context) => {
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
    { headers: { Cookie: cookie } }
  ), { ...env, GOOGLE_PLACES_API_KEY: "test-google-key" });

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "image/jpeg");
  assert.equal((await response.arrayBuffer()).byteLength, 4);
  assert.ok(warnings.some((message) => message.includes("Place photo edge cache unavailable")));
});

test("uncached venue research is limited to five searches per visitor per hour", async (context) => {
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

  for (let index = 0; index < 5; index += 1) {
    const response = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
      method: "POST",
      headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.72" },
      body: JSON.stringify({ query: `Group dinner option ${index}`, area_name: "Soho", lat: 51.513, lng: -0.132 })
    }), recommendationEnv);
    assert.equal(response.status, 200);
  }

  const blocked = await worker.fetch(new Request("https://example.test/api/venue-recommendations", {
    method: "POST",
    headers: { Cookie: cookie, "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.72" },
    body: JSON.stringify({ query: "A sixth uncached search", area_name: "Soho", lat: 51.513, lng: -0.132 })
  }), recommendationEnv);
  assert.equal(blocked.status, 429);
  assert.match((await blocked.json()).detail, /hourly recommendation limit/);
  assert.equal(requests.length, 10);
});
