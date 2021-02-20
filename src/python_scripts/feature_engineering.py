import pandas as pd
from eda_prep import EDA_Prep, data_prep_in_chunks_sql


class Data_Finalization:
    def __init__(self, df):
        self.df = df.copy()
        self.dataset_release_date = pd.to_datetime("2020-3-25 19:13:01")

    # Feature Engineering

    def compute_star_deltas(self):
        self.df["review_stars_minus_user_avg"] = (
            self.df["review_stars"] - self.df["user_avg_stars"]
        )
        self.df["review_stars_minus_business_avg"] = (
            self.df["review_stars"] - self.df["business_avg_stars"]
        )
        self.df["review_stars_v_user_avg_sqr_diff"] = (
            self.df["review_stars"] - self.df["user_avg_stars"]
        ) ** 2
        self.df["review_stars_v_business_avg_sqr_diff"] = (
            self.df["review_stars"] - self.df["business_avg_stars"]
        ) ** 2

    # A lot of work still to do with how time interacts with features.
    # TODO: Get business and user oldest reviews in the dataset.
    # TODO: Use to create more time delta features.
    def compute_time_deltas(self):
        self.df["user_days_active_at_review_time"] = (
            self.df["review_date"] - self.df["user_yelping_since"]
        ).dt.days
        # self.df['business_days_active_at_review_time'] = \
        #     (self.df['review_date'] \
        #      - self.df['business_oldest_review']).dt.days

    def compute_business_feature_comparisons(self):
        self.df["business_checkins_per_review"] = (
            self.df["business_checkin_count"]
            / self.df["business_review_count"]
        )
        self.df["business_checkins_per_review_TD"] = (
            self.df["business_checkin_count_TD"]
            / self.df["business_review_count_TD"]
        )

    def compute_user_feature_comparisons(self):
        # not time adjusted

        # total_ufc per review
        self.df["user_ufc_per_review"] = (
            self.df["user_total_ufc"] / self.df["user_review_count"]
        )
        # total_ufc per years_yelpings
        self.df["user_ufc_per_years_yelping"] = self.df["user_total_ufc"] / (
            (
                (
                    self.dataset_release_date - self.df["user_yelping_since"]
                ).dt.days
            )
            / 365
        )
        # fans per review
        self.df["user_fans_per_review"] = (
            self.df["user_fans"] / self.df["user_review_count"]
        )
        # fans per years_yelping
        self.df["user_fans_per_years_yelping"] = self.df["user_fans"] / (
            (
                (
                    self.dataset_release_date - self.df["user_yelping_since"]
                ).dt.days
            )
            / 365
        )
        # fans per review * total_ufc per review
        self.df["user_fan_per_rev_x_ufc_per_rev"] = (
            self.df["user_fans_per_review"] * self.df["user_ufc_per_review"]
        )

        # time adjusted

        # total_ufc per review
        self.df["user_ufc_per_review_TD"] = (
            self.df["user_total_ufc_TD"] / self.df["user_review_count_TD"]
        )
        # total_ufc per years_yelpings
        self.df["user_ufc_per_years_yelping_TD"] = self.df[
            "user_total_ufc_TD"
        ] / (
            (
                (
                    self.df["review_date"]
                    - (self.df["user_yelping_since"] - pd.Timedelta(days=1))
                ).dt.days
            )
            / 365
        )
        # fans per review
        self.df["user_fans_per_review_TD"] = (
            self.df["user_fans_TD"] / self.df["user_review_count_TD"]
        )
        # fans per years_yelping
        self.df["user_fans_per_years_yelping_TD"] = self.df["user_fans_TD"] / (
            (
                (
                    self.df["review_date"]
                    - (self.df["user_yelping_since"] - pd.Timedelta(days=1))
                ).dt.days
            )
            / 365
        )
        # fans per review * total_ufc per review
        self.df["user_fan_per_rev_x_ufc_per_rev_TD"] = (
            self.df["user_fans_per_review_TD"]
            * self.df["user_ufc_per_review_TD"]
        )

    # Cleaning and Touchup

    def delete_unusable_features(self):
        """
        Delete unneeded feature columns
        and feature columns that could
        cause data leakage.
        """
        columns_to_drop = [
            "user_id",
            "business_id",
            "review_date",
            "business_is_open",
            "business_categories",
            "business_oldest_checkin",
            "business_newest_checkin",
            "business_latitude",
            "business_longitude",
            "user_yelping_since",
        ]
        self.df.drop(labels=columns_to_drop, axis=1, inplace=True)

    # def convert_data_types(self):
    #     pass

    def organize_features(self):
        features_ordered = [
            "review_id",
            "review_stars",
            "review_stars_minus_user_avg",
            "review_stars_minus_business_avg",
            "review_stars_v_user_avg_sqr_diff",
            "review_stars_v_business_avg_sqr_diff",
            "business_avg_stars",
            "business_review_count",
            "business_checkin_count",
            "business_checkins_per_review",
            "business_review_count_TD",
            "business_checkin_count_TD",
            "business_checkins_per_review_TD",
            "user_avg_stars",
            "user_total_ufc",
            "user_review_count",
            "user_friend_count",
            "user_fans",
            "user_compliments",
            "user_elite_count",
            "user_years_since_most_recent_elite",
            "user_days_active_at_review_time",
            "user_ufc_per_review",
            "user_fans_per_review",
            "user_ufc_per_years_yelping",
            "user_fans_per_years_yelping",
            "user_fan_per_rev_x_ufc_per_rev",
            "user_total_ufc_TD",
            "user_review_count_TD",
            "user_friend_count_TD",
            "user_fans_TD",
            "user_compliments_TD",
            "user_elite_count_TD",
            "user_years_since_most_recent_elite_TD",
            "user_ufc_per_review_TD",
            "user_fans_per_review_TD",
            "user_ufc_per_years_yelping_TD",
            "user_fans_per_years_yelping_TD",
            "user_fan_per_rev_x_ufc_per_rev_TD",
            "T1_REG_review_total_ufc",
            "T2_CLS_ufc_>0",
            "T3_CLS_ufc_level",
            "T4_REG_ufc_TD",
            "T5_CLS_ufc_level_TD",
            "T6_REG_ufc_TDBD",
        ]
        self.df = self.df[features_ordered]

    def run_all(self):
        # Feature Engineering
        self.compute_star_deltas()
        self.compute_time_deltas()
        self.compute_business_feature_comparisons()
        self.compute_user_feature_comparisons()
        # Cleaning and Touchup
        self.delete_unusable_features()
        # self.convert_data_types()
        self.organize_features()


if __name__ == "__main__":
    data_prep_in_chunks_sql(100000, "all_features", "cleaned_data", EDA_Prep)
    data_prep_in_chunks_sql(
        100000, "cleaned_data", "non_nlp_model_data", Data_Finalization
    )
