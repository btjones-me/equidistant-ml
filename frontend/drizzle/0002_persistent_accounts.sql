CREATE TABLE `auth_users` (
	`user_id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`first_sign_in_at` integer NOT NULL,
	`last_sign_in_at` integer NOT NULL,
	`sign_in_count` integer NOT NULL
);
