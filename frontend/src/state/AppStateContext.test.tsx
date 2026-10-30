import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AppStateProvider, useAppState, workspaceStorageKey } from "./AppStateContext";

function StateProbe() {
  const {
    friends,
    included,
    mapStyle,
    palette,
    colorScale,
    surfaceOpacity,
    surfaceValueFade,
    suggestionMinDistanceKm,
    setMapStyle,
    setPalette,
    setColorScale,
    setSurfaceOpacity,
    setSurfaceValueFade,
    setSuggestionMinDistanceKm,
    changeFriendCount,
    updateFriend,
    toggleFriend,
    resetWorkspace
  } = useAppState();
  return (
    <div>
      <output aria-label="friend count">{friends.length}</output>
      <output aria-label="first participant">{JSON.stringify(friends[0])}</output>
      <output aria-label="included state">{included.join(",")}</output>
      <output aria-label="map style">{mapStyle}</output>
      <output aria-label="palette state">{palette}</output>
      <output aria-label="low percentile">{colorScale.lowerPercentile}</output>
      <output aria-label="high percentile">{colorScale.upperPercentile}</output>
      <output aria-label="contrast state">{colorScale.contrast}</output>
      <output aria-label="surface opacity">{surfaceOpacity}</output>
      <output aria-label="surface value fade">{surfaceValueFade}</output>
      <output aria-label="suggestion spacing">{suggestionMinDistanceKm}</output>
      <button type="button" onClick={resetWorkspace}>Reset workspace</button>
      <button type="button" onClick={() => toggleFriend(1)}>Toggle Sam</button>
      <button type="button" onClick={() => setMapStyle("voyager")}>Street map</button>
      <button type="button" onClick={() => setPalette("green-red")}>Green red</button>
      <button type="button" onClick={() => setColorScale((current) => ({ ...current, contrast: 1.4 }))}>Contrast</button>
      <button type="button" onClick={() => setSurfaceOpacity(0.64)}>Opacity</button>
      <button type="button" onClick={() => setSurfaceValueFade(0.72)}>Value fade</button>
      <button type="button" onClick={() => setSuggestionMinDistanceKm(4.5)}>Spacing</button>
      <button type="button" onClick={() => changeFriendCount(1)}>One friend</button>
      <button type="button" onClick={() => updateFriend(0, { name: "Profile A participant", lat: 51.5, lng: -0.15 })}>Edit participant</button>
    </div>
  );
}

function renderProbe(accountId = "account-a") {
  return render(<AppStateProvider key={accountId} accountId={accountId}><StateProbe /></AppStateProvider>);
}

describe("shared application state", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(cleanup);

  it("starts with a useful three-person workspace", () => {
    renderProbe();
    expect(screen.getByLabelText("friend count")).toHaveTextContent("3");
    expect(screen.getByLabelText("included state")).toHaveTextContent("true,true,true");
    expect(screen.getByLabelText("low percentile")).toHaveTextContent("1");
    expect(screen.getByLabelText("high percentile")).toHaveTextContent("58");
    expect(screen.getByLabelText("contrast state")).toHaveTextContent("1");
    expect(screen.getByLabelText("surface opacity")).toHaveTextContent("0.75");
    expect(screen.getByLabelText("surface value fade")).toHaveTextContent("1");
  });

  it("persists selections and colour settings across mode remounts", () => {
    const first = renderProbe();
    fireEvent.click(screen.getByRole("button", { name: "Toggle Sam" }));
    fireEvent.click(screen.getByRole("button", { name: "Street map" }));
    fireEvent.click(screen.getByRole("button", { name: "Green red" }));
    fireEvent.click(screen.getByRole("button", { name: "Contrast" }));
    fireEvent.click(screen.getByRole("button", { name: "Opacity" }));
    fireEvent.click(screen.getByRole("button", { name: "Value fade" }));
    fireEvent.click(screen.getByRole("button", { name: "Spacing" }));
    first.unmount();

    renderProbe();
    expect(screen.getByLabelText("included state")).toHaveTextContent("true,false,true");
    expect(screen.getByLabelText("map style")).toHaveTextContent("voyager");
    expect(screen.getByLabelText("palette state")).toHaveTextContent("green-red");
    expect(screen.getByLabelText("contrast state")).toHaveTextContent("1.4");
    expect(screen.getByLabelText("surface opacity")).toHaveTextContent("0.64");
    expect(screen.getByLabelText("surface value fade")).toHaveTextContent("0.72");
    expect(screen.getByLabelText("suggestion spacing")).toHaveTextContent("4.5");
  });

  it("supports a one-person reachability surface", () => {
    renderProbe();
    fireEvent.click(screen.getByRole("button", { name: "One friend" }));
    expect(screen.getByLabelText("friend count")).toHaveTextContent("1");
    expect(screen.getByLabelText("included state")).toHaveTextContent("true");
  });

  it("does not copy one browser's saved participants into separate browser storage", () => {
    const storageKey = workspaceStorageKey("account-a");
    const first = renderProbe();
    fireEvent.click(screen.getByRole("button", { name: "Edit participant" }));
    const profileA = window.localStorage.getItem(storageKey)!;
    expect(JSON.parse(profileA).friends[0].name).toBe("Profile A participant");
    first.unmount();

    // A separate browser starts with its own empty origin storage, while the
    // same module remains loaded here to catch accidental mutable defaults.
    window.localStorage.clear();
    const second = renderProbe();
    expect(screen.getByLabelText("first participant")).not.toHaveTextContent("Profile A participant");
    expect(JSON.parse(window.localStorage.getItem(storageKey)!).friends[0].lat).not.toBe(51.5);
    second.unmount();

    window.localStorage.clear();
    window.localStorage.setItem(storageKey, profileA);
    renderProbe();
    expect(screen.getByLabelText("first participant")).toHaveTextContent("Profile A participant");
    expect(screen.getByLabelText("first participant")).toHaveTextContent('"lat":51.5');
  });
  it("keeps Google accounts separate in the same browser without adopting unowned legacy data", () => {
    const legacy = JSON.stringify({ friends: [{ id: "legacy", name: "Unowned participant", lat: 51.6, lng: -0.1 }] });
    window.localStorage.setItem("equidistant:workspace:v2", legacy);
    const view = render(<AppStateProvider accountId="account-a"><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).not.toHaveTextContent("Unowned participant");
    fireEvent.click(screen.getByRole("button", { name: "Edit participant" }));
    view.rerender(<AppStateProvider accountId="account-b"><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).not.toHaveTextContent("Profile A participant");
    view.rerender(<AppStateProvider accountId="account-a"><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).toHaveTextContent("Profile A participant");
    expect(window.localStorage.getItem("equidistant:workspace:v2")).toBe(legacy);
  });

  it("keeps the 50 pubs workspace separate from the ordinary planner for the same account", () => {
    const preset = [{ id: "pair-one", name: "First pair", lat: 51.53, lng: -0.12, locationLabel: "Example station" }];
    const view = render(<AppStateProvider accountId="account-a"><StateProbe /></AppStateProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Edit participant" }));
    const mainSaved = window.localStorage.getItem(workspaceStorageKey("account-a"));
    view.rerender(<AppStateProvider accountId="account-a" workspace="50pubs" initialFriends={preset}><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).toHaveTextContent("First pair");
    expect(screen.getByLabelText("friend count")).toHaveTextContent("1");
    expect(window.localStorage.getItem(workspaceStorageKey("account-a"))).toBe(mainSaved);
    expect(JSON.parse(window.localStorage.getItem(workspaceStorageKey("account-a", "50pubs"))!).friends).toEqual(preset);
    view.rerender(<AppStateProvider accountId="account-a"><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).toHaveTextContent("Profile A participant");
  });

  it("preserves saved pub-workspace edits on remount and resets to the private preset", () => {
    const preset = [
      { id: "pair-one", name: "First pair", lat: 51.53, lng: -0.12, locationLabel: "Example station" },
      { id: "pair-two", name: "Second pair", lat: 51.54, lng: -0.13, locationLabel: "Other station" }
    ];
    const make = () => <AppStateProvider accountId="account-a" workspace="50pubs" initialFriends={preset}><StateProbe /></AppStateProvider>;
    const first = render(make());
    fireEvent.click(screen.getByRole("button", { name: "Edit participant" }));
    fireEvent.click(screen.getByRole("button", { name: "Toggle Sam" }));
    first.unmount();
    render(make());
    expect(screen.getByLabelText("first participant")).toHaveTextContent("Profile A participant");
    expect(screen.getByLabelText("included state")).toHaveTextContent("true,false");
    fireEvent.click(screen.getByRole("button", { name: "Reset workspace" }));
    expect(screen.getByLabelText("first participant")).toHaveTextContent("First pair");
    expect(screen.getByLabelText("included state")).toHaveTextContent("true,true");
    expect(JSON.parse(window.localStorage.getItem(workspaceStorageKey("account-a", "50pubs"))!).friends).toEqual(preset);
    expect(preset[0].name).toBe("First pair");
  });

  it("does not expose a saved owner pub preset to another account", () => {
    const preset = [{ id: "private-pair", name: "Private test pair", lat: 51.53, lng: -0.12 }];
    const view = render(<AppStateProvider accountId="owner" workspace="50pubs" initialFriends={preset}><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).toHaveTextContent("Private test pair");
    view.rerender(<AppStateProvider accountId="different-account" workspace="50pubs" initialFriends={[]}><StateProbe /></AppStateProvider>);
    expect(screen.getByLabelText("first participant")).not.toHaveTextContent("Private test pair");
    expect(window.localStorage.getItem(workspaceStorageKey("different-account", "50pubs"))).not.toContain("Private test pair");
  });

});
