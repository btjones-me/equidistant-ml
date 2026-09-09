CREATE TABLE `auth_flows` (
	`flow_hash` text PRIMARY KEY NOT NULL,
	`state` text NOT NULL,
	`nonce` text NOT NULL,
	`verifier` text NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `auth_sessions` (
	`token_hash` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`email` text NOT NULL,
	`name` text NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `auth_sessions_expiry` ON `auth_sessions` (`expires_at`);--> statement-breakpoint
CREATE TABLE `recommendation_events` (
	`request_id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `recommendation_events_user_time` ON `recommendation_events` (`user_id`,`created_at`);