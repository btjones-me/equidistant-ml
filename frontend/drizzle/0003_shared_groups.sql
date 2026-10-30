CREATE TABLE `group_availability` (
	`group_id` text NOT NULL,
	`person_id` text NOT NULL,
	`date` text NOT NULL,
	`status` text NOT NULL,
	`updated_at` integer NOT NULL,
	`updated_by` text NOT NULL,
	PRIMARY KEY(`group_id`, `person_id`, `date`)
);
--> statement-breakpoint
CREATE INDEX `group_availability_month` ON `group_availability` (`group_id`,`date`);--> statement-breakpoint
CREATE TABLE `group_invites` (
	`token_hash` text PRIMARY KEY NOT NULL,
	`group_id` text NOT NULL,
	`created_by` text NOT NULL,
	`created_at` integer NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `group_invites_expiry` ON `group_invites` (`expires_at`);--> statement-breakpoint
CREATE INDEX `group_invites_group` ON `group_invites` (`group_id`);--> statement-breakpoint
CREATE TABLE `group_members` (
	`group_id` text NOT NULL,
	`user_id` text NOT NULL,
	`joined_at` integer NOT NULL,
	PRIMARY KEY(`group_id`, `user_id`)
);
--> statement-breakpoint
CREATE INDEX `group_members_user` ON `group_members` (`user_id`);--> statement-breakpoint
CREATE TABLE `group_origins` (
	`group_id` text NOT NULL,
	`origin_id` text NOT NULL,
	`name` text NOT NULL,
	`location_label` text NOT NULL,
	`lat` real NOT NULL,
	`lng` real NOT NULL,
	`position` integer NOT NULL,
	PRIMARY KEY(`group_id`, `origin_id`)
);
--> statement-breakpoint
CREATE TABLE `group_people` (
	`group_id` text NOT NULL,
	`person_id` text NOT NULL,
	`name` text NOT NULL,
	`origin_id` text,
	`position` integer NOT NULL,
	`deleted_at` integer,
	PRIMARY KEY(`group_id`, `person_id`)
);
--> statement-breakpoint
CREATE INDEX `group_people_origin` ON `group_people` (`group_id`,`origin_id`);--> statement-breakpoint
CREATE TABLE `group_rate_limits` (
	`scope` text PRIMARY KEY NOT NULL,
	`window_started_at` integer NOT NULL,
	`requests` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `planning_groups` (
	`group_id` text PRIMARY KEY NOT NULL,
	`name` text NOT NULL,
	`created_by` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `planning_groups_creator` ON `planning_groups` (`created_by`);