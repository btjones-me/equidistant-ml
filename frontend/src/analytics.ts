export const GA_MEASUREMENT_ID = "G-2DH3C5NTDZ";

const SCRIPT_ID = "equidistant-google-analytics";

type GtagArguments = [command: string, ...values: unknown[]];

declare global {
  interface Window {
    dataLayer?: GtagArguments[];
    gtag?: (...args: GtagArguments) => void;
  }
}

export function initializeAnalytics() {
  if (typeof document === "undefined" || document.getElementById(SCRIPT_ID)) {
    return;
  }

  window.dataLayer = window.dataLayer || [];
  window.gtag = (...args: GtagArguments) => window.dataLayer?.push(args);

  // Keep storage disabled by default. GA still receives privacy-preserving,
  // cookieless measurement pings without setting advertising or analytics cookies.
  window.gtag("consent", "default", {
    ad_storage: "denied",
    ad_user_data: "denied",
    ad_personalization: "denied",
    analytics_storage: "denied"
  });
  window.gtag("js", new Date());
  window.gtag("config", GA_MEASUREMENT_ID, { anonymize_ip: true });

  const script = document.createElement("script");
  script.id = SCRIPT_ID;
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script);
}
