
/* Renaming user table to users to avoid Postgres reserved words issue. */

ALTER TABLE "user"
RENAME TO users
;

/* Adding primary keys in Postgres after import from Json file. */
/* No unique identifier (primary key) for tips table. */

ALTER TABLE business 
ADD PRIMARY KEY (business_id)
;

ALTER TABLE users 
ADD PRIMARY KEY (user_id)
;

ALTER TABLE review 
ADD PRIMARY KEY (review_id)
;

ALTER TABLE checkin 
ADD PRIMARY KEY (business_id)
;
