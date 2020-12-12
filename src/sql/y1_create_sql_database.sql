
/* 
WARNING: Final table is ~85GB. 
*/

/* 
Reduce all businesses to just restaurants. 
*/

SELECT *
INTO restaurants
FROM business
WHERE categories LIKE '%Restaurant%'
;

/* 
Rename column with "." in it to avoid future errors. 
*/

ALTER TABLE restaurants
RENAME COLUMN "attributes.RestaurantsPriceRange2" TO price_range
;

/* 
Combine reviews and restaurants.
Only return reviews from restaurants that have values in both tables.
Rename tables to avoid duplicates and confusion later on. 
*/

SELECT
review.review_id,
review.user_id,
review.business_id,
review.stars              AS review_stars,
review.date               AS review_date,
review.text               AS review_text,
review.useful             AS review_useful,
review.funny              AS review_funny,
review.cool               AS review_cool,
restaurants.name          AS restaurant_name,
restaurants.address       AS restaurant_address,
restaurants.city          AS restaurant_city,
restaurants.state         AS restaurant_state,
restaurants.postal_code   AS restaurant_postal_code,
restaurants.latitude      AS restaurant_latitude,
restaurants.longitude     AS restaurant_longitude,
restaurants.stars         AS restaurant_overall_stars,
restaurants.review_count  AS restaurant_review_count,
restaurants.is_open       AS restaurant_is_open,
restaurants.categories    AS restaurant_categories,
restaurants.price_range   AS restaurant_price_range
INTO restaurant_reviews
FROM review
INNER JOIN restaurants
ON review.business_id = restaurants.business_id
;

/* 
Combining restaurant reviews with user and business checkin data.
Only return records that have review, business, and user data.
Return records whether or not they have checkin data.
*/

SELECT
restaurant_reviews.review_id,
restaurant_reviews.user_id,
restaurant_reviews.business_id,
restaurant_reviews.review_stars,
restaurant_reviews.review_date,
restaurant_reviews.review_text,
restaurant_reviews.review_useful,
restaurant_reviews.review_funny,
restaurant_reviews.review_cool,
restaurant_reviews.restaurant_name,
restaurant_reviews.restaurant_address,
restaurant_reviews.restaurant_city,
restaurant_reviews.restaurant_state,
restaurant_reviews.restaurant_postal_code,
restaurant_reviews.restaurant_latitude,
restaurant_reviews.restaurant_longitude,
restaurant_reviews.restaurant_overall_stars,
restaurant_reviews.restaurant_review_count,
restaurant_reviews.restaurant_is_open,
restaurant_reviews.restaurant_categories,
restaurant_reviews.restaurant_price_range,
users.name               AS user_name,
users.review_count       AS user_review_count,
users.yelping_since      AS user_yelping_since,
users.useful             AS user_useful,
users.funny              AS user_funny,
users.cool               AS user_cool,
users.elite              AS user_elite,
users.friends            AS user_friends,
users.fans               AS user_fans,
users.average_stars      AS user_average_stars_given,
users.compliment_hot     AS user_compliment_hot,
users.compliment_more    AS user_compliment_more,
users.compliment_profile AS user_compliment_profile,
users.compliment_cute    AS user_compliment_cute,
users.compliment_list    AS user_compliment_list,
users.compliment_note    AS user_compliment_note,
users.compliment_plain   AS user_compliment_plain,
users.compliment_cool    AS user_compliment_cool,
users.compliment_funny   AS user_compliment_funny,
users.compliment_writer  AS user_compliment_writer,
users.compliment_photos  AS user_compliment_photos,
checkin.date             AS restaurant_checkins
INTO restaurant_reviews_final
FROM restaurant_reviews
INNER JOIN users
ON restaurant_reviews.user_id = users.user_id
JOIN checkin
ON restaurant_reviews.business_id = checkin.business_id
;
