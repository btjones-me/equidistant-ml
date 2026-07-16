import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "leaflet/dist/leaflet.css";
import "./styles.css";
import App from "./App";
import { initializeAnalytics } from "./analytics";

initializeAnalytics();

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
