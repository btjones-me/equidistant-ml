import { lazy, Suspense, useEffect, useState } from "react";
import ProductMode from "./ProductMode";
import { AppStateProvider } from "./state/AppStateContext";
import { type AccountSession, setActiveAccount, signOut } from "./lib/account";
import type { Friend, ExperienceMode } from "./types";

const fiftyPubsMode = /^\/50pubs\/?$/.test(window.location.pathname);
const DeveloperMode = lazy(() => import("./DeveloperMode"));

function AppShell({ session }: { session: AccountSession }) {
  const [mode, setMode] = useState<ExperienceMode>("explore");
  return mode === "developer" && session.permissions.debug ? (
    <Suspense fallback={<p role="status">Opening diagnostics…</p>}><DeveloperMode onExit={() => setMode("explore")} /></Suspense>
  ) : (
    <ProductMode fiftyPubsMode={fiftyPubsMode} accountEmail={session.user.email} onSignOut={signOut}
      onDeveloperMode={session.permissions.debug ? () => setMode("developer") : undefined} />
  );
}

export default function App() {
  const [session, setSession] = useState<AccountSession | null>(null);
  const [error, setError] = useState(false);
  const [preset, setPreset] = useState<Friend[]>([]);
  useEffect(() => {
    let active = true;
    let sequence = 0;
    async function refresh() {
      const requestSequence = ++sequence;
      try {
        const response = await fetch("/api/session", { cache: "no-store" });
        if (!active || requestSequence !== sequence) return;
        if (response.status === 401) { window.location.replace(fiftyPubsMode ? "/50pubs" : "/"); return; }
        if (!response.ok) throw new Error("Session unavailable");
        const next = await response.json() as AccountSession;
        if (!active || requestSequence !== sequence) return;
        if (fiftyPubsMode) {
          const presetResponse = await fetch("/api/50pubs/preset", { cache: "no-store", headers: { "X-Equidistant-Account": next.user.id } });
          if (!presetResponse.ok) throw new Error("Preset unavailable");
          const result = await presetResponse.json() as {friends: Friend[]};
          if (!active || requestSequence !== sequence) return;
          setPreset(result.friends);
        }
        setActiveAccount(next.user.id);
        setSession(next);
        setError(false);
      } catch {
        if (active && requestSequence === sequence) { setActiveAccount(""); setSession(null); setError(true); }
      }
    }
    function resume() {
      if (document.visibilityState === "hidden") return;
      // Recheck shared cookies before displaying a workspace restored from another tab/session.
      setSession(null);
      setActiveAccount("");
      void refresh();
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), 60_000);
    window.addEventListener("pageshow", resume);
    window.addEventListener("focus", resume);
    document.addEventListener("visibilitychange", resume);
    return () => {
      active = false;
      window.clearInterval(timer);
      window.removeEventListener("pageshow", resume);
      window.removeEventListener("focus", resume);
      document.removeEventListener("visibilitychange", resume);
    };
  }, []);
  if (!session) return <main className="account-loading"><p role={error ? "alert" : "status"}>{error ? "We couldn’t check your sign-in. Please reload to try again." : "Opening your workspace…"}</p></main>;
  return (
    <AppStateProvider key={session.user.id} accountId={session.user.id} workspace={fiftyPubsMode ? "50pubs" : "main"} initialFriends={preset}>
      <AppShell session={session} />
    </AppStateProvider>
  );
}
