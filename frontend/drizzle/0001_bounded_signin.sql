CREATE TABLE `auth_rate_limits` (
	`client_key` text PRIMARY KEY NOT NULL,
	`window_started_at` integer NOT NULL,
	`attempts` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `auth_flows_expiry` ON `auth_flows` (`expires_at`);