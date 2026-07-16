import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AppStateProvider, useAppState } from "./AppStateContext";

function StateProbe() {
  const {
    friends,
    included,
    palette,
    colorScale,
    surfaceOpacity,
    suggestionMinDistanceKm,
    setPalette,
    setColorScale,
    setSurfaceOpacity,
    setSuggestionMinDistanceKm,
    changeFriendCount,
    toggleFriend
  } = useAppState();
  return (
    <div>
      <output aria-label="friend count">{friends.length}</output>
      <output aria-label="included state">{included.join(",")}</output>
      <output aria-label="palette state">{palette}</output>
      <output aria-label="contrast state">{colorScale.contrast}</output>
      <output aria-label="surface opacity">{surfaceOpacity}</output>
      <output aria-label="suggestion spacing">{suggestionMinDistanceKm}</output>
      <button type="button" onClick={() => toggleFriend(1)}>Toggle Sam</button>
      <button type="button" onClick={() => setPalette("green-red")}>Green red</button>
      <button type="button" onClick={() => setColorScale((current) => ({ ...current, contrast: 1.4 }))}>Contrast</button>
      <button type="button" onClick={() => setSurfaceOpacity(0.64)}>Opacity</button>
      <button type="button" onClick={() => setSuggestionMinDistanceKm(4.5)}>Spacing</button>
      <button type="button" onClick={() => changeFriendCount(1)}>One friend</button>
    </div>
  );
}

function renderProbe() {
  return render(<AppStateProvider><StateProbe /></AppStateProvider>);
}

describe("shared application state", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(cleanup);

  it("starts with a useful three-person workspace", () => {
    renderProbe();
    expect(screen.getByLabelText("friend count")).toHaveTextContent("3");
    expect(screen.getByLabelText("included state")).toHaveTextContent("true,true,true");
  });

  it("persists selections and colour settings across mode remounts", () => {
    const first = renderProbe();
    fireEvent.click(screen.getByRole("button", { name: "Toggle Sam" }));
    fireEvent.click(screen.getByRole("button", { name: "Green red" }));
    fireEvent.click(screen.getByRole("button", { name: "Contrast" }));
    fireEvent.click(screen.getByRole("button", { name: "Opacity" }));
    fireEvent.click(screen.getByRole("button", { name: "Spacing" }));
    first.unmount();

    renderProbe();
    expect(screen.getByLabelText("included state")).toHaveTextContent("true,false,true");
    expect(screen.getByLabelText("palette state")).toHaveTextContent("green-red");
    expect(screen.getByLabelText("contrast state")).toHaveTextContent("1.4");
    expect(screen.getByLabelText("surface opacity")).toHaveTextContent("0.64");
    expect(screen.getByLabelText("suggestion spacing")).toHaveTextContent("4.5");
  });

  it("supports a one-person reachability surface", () => {
    renderProbe();
    fireEvent.click(screen.getByRole("button", { name: "One friend" }));
    expect(screen.getByLabelText("friend count")).toHaveTextContent("1");
    expect(screen.getByLabelText("included state")).toHaveTextContent("true");
  });
});
