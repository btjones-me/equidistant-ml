import { index, integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const authFlows = sqliteTable("auth_flows", {
  flowHash: text("flow_hash").primaryKey(),
  state: text("state").notNull(),
  nonce: text("nonce").notNull(),
  verifier: text("verifier").notNull(),
  expiresAt: integer("expires_at").notNull()
}, (table) => [index("auth_flows_expiry").on(table.expiresAt)]);

export const authRateLimits = sqliteTable("auth_rate_limits", {
  clientKey: text("client_key").primaryKey(),
  windowStartedAt: integer("window_started_at").notNull(),
  attempts: integer("attempts").notNull()
});

export const authSessions = sqliteTable("auth_sessions", {
  tokenHash: text("token_hash").primaryKey(),
  userId: text("user_id").notNull(),
  email: text("email").notNull(),
  name: text("name").notNull(),
  expiresAt: integer("expires_at").notNull()
}, (table) => [index("auth_sessions_expiry").on(table.expiresAt)]);

export const authUsers = sqliteTable("auth_users", {
  userId: text("user_id").primaryKey(),
  email: text("email").notNull(),
  firstSignInAt: integer("first_sign_in_at").notNull(),
  lastSignInAt: integer("last_sign_in_at").notNull(),
  signInCount: integer("sign_in_count").notNull()
});

export const recommendationEvents = sqliteTable("recommendation_events", {
  requestId: text("request_id").primaryKey(),
  userId: text("user_id").notNull(),
  createdAt: integer("created_at").notNull()
}, (table) => [index("recommendation_events_user_time").on(table.userId, table.createdAt)]);
