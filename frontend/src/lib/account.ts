export type AccountSession = {
  user: { id: string; email: string; name: string };
  permissions: { debug: boolean; liveTravelTime: boolean; unlimitedRecommendations: boolean };
};

let accountId = "";
export const activeAccountId = () => accountId;
export function setActiveAccount(id: string) { accountId = id; }

export function accountFetch(url: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  if (accountId) headers.set("X-Equidistant-Account", accountId);
  return fetch(url, { ...init, headers });
}

export function signOut() {
  const form = document.createElement("form");
  form.method = "post";
  form.action = "/auth/logout";
  document.body.appendChild(form);
  form.submit();
}
