import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { useAppState } from "./state/AppStateContext";

vi.mock("./ProductMode", () => ({ default: ({ onDeveloperMode, accountEmail }: { onDeveloperMode?: () => void; accountEmail: string }) => {
  const { friends, updateFriend } = useAppState();
  return <div><p>{accountEmail}</p><p>{friends[0].name}</p>
    <button onClick={() => updateFriend(0, { name: "Private account A entry" })}>Edit</button>
    {onDeveloperMode ? <button onClick={onDeveloperMode}>Diagnostics</button> : null}</div>;
} }));
vi.mock("./DeveloperMode", () => ({ default: () => <p>Restricted diagnostics</p> }));
const session = (id: string, debug: boolean) => ({ user: { id, email: `${id}@example.test`, name: id }, permissions: { debug, liveTravelTime: debug, unlimitedRecommendations: debug } });
beforeEach(() => window.localStorage.clear());
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("ordinary users cannot open diagnostics", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(session("ordinary", false))));
  render(<App />);
  await screen.findByText("ordinary@example.test");
  expect(screen.queryByRole("button", { name: "Diagnostics" })).not.toBeInTheDocument();
});

it("refreshing an account after another tab signs in replaces the workspace and permissions", async () => {
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(Response.json(session("account-a", true))));
  vi.stubGlobal("fetch", fetch);
  render(<App />);
  await screen.findByText("account-a@example.test");
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  expect(screen.getByText("Private account A entry")).toBeInTheDocument();
  fetch.mockImplementation(() => Promise.resolve(Response.json(session("account-b", false))));
  fireEvent.focus(window);
  await screen.findByText("account-b@example.test");
  expect(screen.queryByText("Private account A entry")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Diagnostics" })).not.toBeInTheDocument();
  fetch.mockImplementation(() => Promise.resolve(Response.json(session("account-a", true))));
  fireEvent.focus(window);
  await waitFor(() => expect(screen.getByText("Private account A entry")).toBeInTheDocument());
});
