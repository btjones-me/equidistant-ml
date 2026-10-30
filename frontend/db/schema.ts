import { index, integer, primaryKey, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

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


export const planningGroups = sqliteTable("planning_groups", {
  id: text("group_id").primaryKey(),
  name: text("name").notNull(),
  createdBy: text("created_by").notNull(),
  createdAt: integer("created_at").notNull()
}, table => [index("planning_groups_creator").on(table.createdBy)]);

export const groupMembers = sqliteTable("group_members", {
  groupId: text("group_id").notNull(),
  userId: text("user_id").notNull(),
  joinedAt: integer("joined_at").notNull()
}, table => [primaryKey({ columns: [table.groupId, table.userId] }), index("group_members_user").on(table.userId)]);

export const groupOrigins = sqliteTable("group_origins", {
  groupId: text("group_id").notNull(),
  id: text("origin_id").notNull(),
  name: text("name").notNull(),
  locationLabel: text("location_label").notNull(),
  lat: real("lat").notNull(),
  lng: real("lng").notNull(),
  position: integer("position").notNull()
}, table => [primaryKey({ columns: [table.groupId, table.id] })]);

export const groupPeople = sqliteTable("group_people", {
  groupId: text("group_id").notNull(),
  id: text("person_id").notNull(),
  name: text("name").notNull(),
  originId: text("origin_id"),
  position: integer("position").notNull(),
  deletedAt: integer("deleted_at")
}, table => [primaryKey({ columns: [table.groupId, table.id] }), index("group_people_origin").on(table.groupId, table.originId)]);

export const groupAvailability = sqliteTable("group_availability", {
  groupId: text("group_id").notNull(),
  personId: text("person_id").notNull(),
  date: text("date").notNull(),
  status: text("status").notNull(),
  updatedAt: integer("updated_at").notNull(),
  updatedBy: text("updated_by").notNull()
}, table => [primaryKey({ columns: [table.groupId, table.personId, table.date] }), index("group_availability_month").on(table.groupId, table.date)]);

export const groupInvites = sqliteTable("group_invites", {
  tokenHash: text("token_hash").primaryKey(),
  groupId: text("group_id").notNull(),
  createdBy: text("created_by").notNull(),
  createdAt: integer("created_at").notNull(),
  expiresAt: integer("expires_at").notNull()
}, table => [index("group_invites_expiry").on(table.expiresAt), index("group_invites_group").on(table.groupId)]);

export const groupRateLimits = sqliteTable("group_rate_limits", {
  scope: text("scope").primaryKey(),
  windowStartedAt: integer("window_started_at").notNull(),
  requests: integer("requests").notNull()
});
