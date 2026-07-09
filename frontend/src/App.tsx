import { useState } from "react";
import DeveloperMode from "./DeveloperMode";
import ProductMode from "./ProductMode";
import { AppStateProvider } from "./state/AppStateContext";
import type { ExperienceMode } from "./types";

function AppShell() {
  const [mode, setMode] = useState<ExperienceMode>("explore");
  return mode === "developer" ? (
    <DeveloperMode onExit={() => setMode("explore")} />
  ) : (
    <ProductMode onDeveloperMode={() => setMode("developer")} />
  );
}

export default function App() {
  return (
    <AppStateProvider>
      <AppShell />
    </AppStateProvider>
  );
}
