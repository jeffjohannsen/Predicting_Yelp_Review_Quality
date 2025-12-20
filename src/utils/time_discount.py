"""
Time Discounting Utilities for Yelp Review Features

This module provides time discounting functionality to adjust feature values
based on when a review was created, accounting for the temporal bias in the
dataset where newer reviews have had less time to accumulate votes and
user/business metrics have grown over time.

The core problem: Dataset snapshot is from 2020, but contains reviews from
2004-2020. Features like vote counts, user stats, and business metrics
accumulate over time, so we must "rewind" them to their approximate values
at the time each review was created.

Key Concepts:
    - Dataset Release Date: 2020-03-25 (when snapshot was taken)
    - Yelp Founding Date: 2004-07-01 (used for business features)
    - Time Discount Formula: (current_value / days_total) * days_at_review_time

Classes:
    TimeDiscountCalculator: Main class containing all time discount methods

Example:
    >>> from datetime import datetime
    >>> calc = TimeDiscountCalculator()
    >>>
    >>> # Discount a review's target votes
    >>> review_date = datetime(2018, 6, 15)
    >>> total_votes = 42
    >>> discounted = calc.target_time_discount(total_votes, review_date)
    >>>
    >>> # Discount user features
    >>> user_since = datetime(2010, 1, 1)
    >>> user_review_count = 150
    >>> discounted_count = calc.user_time_discount(
    ...     user_review_count, user_since, review_date
    ... )
"""

from datetime import datetime, timedelta
from typing import Optional, Union


class TimeDiscountCalculator:
    """
    Calculator for time-discounting review features to account for temporal bias.

    This class provides methods to adjust feature values based on review creation
    time, ensuring that features represent approximate values at the time the
    review was written rather than at dataset release time.

    Attributes:
        dataset_release_date (datetime): Date when dataset snapshot was taken
        yelp_founding_date (datetime): Date Yelp was founded (for business features)

    Methods:
        target_time_discount: Discount review vote counts (useful/funny/cool)
        user_time_discount: Discount user count features (reviews, fans, friends)
        business_time_discount: Discount business count features (reviews, checkins)
        count_elite_td: Count elite years before review date
        years_since_elite_td: Years since most recent elite status before review
        usefulness_level: Categorize usefulness into zero/low/medium/high
        create_all_targets: Create all 6 target variations (T1-T6)
    """

    def __init__(
        self,
        dataset_release_date: Optional[datetime] = None,
        yelp_founding_date: Optional[datetime] = None,
    ):
        """
        Initialize TimeDiscountCalculator with reference dates.

        Args:
            dataset_release_date: Dataset snapshot date. Defaults to 2020-03-25.
            yelp_founding_date: Yelp company founding date. Defaults to 2004-07-01.
        """
        self.dataset_release_date = dataset_release_date or datetime(
            2020, 3, 25, 19, 13, 1
        )
        self.yelp_founding_date = yelp_founding_date or datetime(2004, 7, 1, 0, 0, 0)

    def target_time_discount(
        self, ufc_total: Union[int, float], review_date: datetime
    ) -> float:
        """
        Time-discount review target votes (useful + funny + cool).

        Adjusts vote counts to account for the time reviews have had to accumulate
        votes. Newer reviews have less time to get votes, so raw counts are biased.

        Formula: (total_votes / days_since_review) * 365

        This normalizes votes to an "annual rate" as if all reviews were 1 year old.

        Args:
            ufc_total: Total votes (useful + funny + cool)
            review_date: Date when review was created

        Returns:
            Time-discounted vote count (annualized rate)

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> review_date = datetime(2019, 6, 1)  # 10 months before snapshot
            >>> votes = 20
            >>> calc.target_time_discount(votes, review_date)
            # Returns ~24 (20 votes in 10 months → ~24 votes/year rate)
        """
        days_since_review = (self.dataset_release_date - review_date).days

        # Avoid division by zero for same-day reviews
        if days_since_review <= 0:
            days_since_review = 1

        return (ufc_total / days_since_review) * 365

    def user_time_discount(
        self,
        count_feature: Union[int, float],
        user_yelping_since: datetime,
        review_date: datetime,
    ) -> float:
        """
        Time-discount user count features to review creation time.

        Adjusts user metrics (review count, fans, friends, votes, compliments)
        to estimate their values when the review was written, not at dataset
        release time.

        Formula: (current_value / days_user_active_total) * days_user_active_at_review

        Args:
            count_feature: User count value at dataset release (reviews, fans, etc.)
            user_yelping_since: Date user joined Yelp
            review_date: Date when review was created

        Returns:
            Estimated count value at review creation time

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> user_since = datetime(2010, 1, 1)
            >>> review_date = datetime(2015, 6, 1)  # Halfway through user tenure
            >>> current_reviews = 200  # At 2020
            >>> calc.user_time_discount(current_reviews, user_since, review_date)
            # Returns ~100 (user had written ~half their reviews by 2015)

        Note:
            Subtracts 1 day from yelping_since to handle edge cases where
            review_date == yelping_since (user's first review).
        """
        # Subtract 1 day to handle users reviewing on join date
        user_start = user_yelping_since - timedelta(days=1)

        days_total = (self.dataset_release_date - user_start).days
        days_at_review = (review_date - user_start).days

        # Avoid division by zero
        if days_total <= 0:
            days_total = 1
        if days_at_review <= 0:
            days_at_review = 1

        return (count_feature / days_total) * days_at_review

    def business_time_discount(
        self, count_feature: Union[int, float], review_date: datetime
    ) -> float:
        """
        Time-discount business count features to review creation time.

        Adjusts business metrics (review count, checkin count) to estimate
        their values when the review was written. Uses Yelp founding date
        as business start date since we don't have individual business
        creation dates.

        Formula: (current_value / days_since_yelp_founding) * days_from_founding_to_review

        Args:
            count_feature: Business count value at dataset release
            review_date: Date when review was created

        Returns:
            Estimated count value at review creation time

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> review_date = datetime(2015, 1, 1)
            >>> current_reviews = 500  # Business has 500 reviews in 2020
            >>> calc.business_time_discount(current_reviews, review_date)
            # Returns discounted count based on time from 2004 to 2015 vs 2004 to 2020

        Note:
            This is an approximation since we don't have actual business
            creation dates. Documented as a known limitation in project docs.
        """
        days_total = (self.dataset_release_date - self.yelp_founding_date).days
        days_at_review = (review_date - self.yelp_founding_date).days

        # Avoid division by zero
        if days_total <= 0:
            days_total = 1
        if days_at_review <= 0:
            days_at_review = 1

        return (count_feature / days_total) * days_at_review

    def count_elite_td(self, user_elite: Optional[str], review_date: datetime) -> int:
        """
        Count elite years achieved before review date.

        Users can have elite status in multiple years (e.g., "2015,2016,2017").
        This counts how many elite years they had accumulated by review date.

        Args:
            user_elite: Comma-separated string of elite years (e.g., "2015,2016,2017")
            review_date: Date when review was created

        Returns:
            Number of elite years before or at review date

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> elite_str = "2010,2012,2015,2018"
            >>> review_date = datetime(2016, 6, 1)
            >>> calc.count_elite_td(elite_str, review_date)
            # Returns 3 (2010, 2012, 2015 were before 2016)
        """
        if user_elite in ["None", None, ""]:
            return 0

        try:
            elite_years = list(map(int, user_elite.split(",")))
            elite_before_review = [
                year for year in elite_years if year <= review_date.year
            ]
            return len(elite_before_review)
        except (ValueError, AttributeError):
            return 0

    def years_since_elite_td(
        self, user_elite: Optional[str], review_date: datetime
    ) -> int:
        """
        Calculate years since most recent elite status before review date.

        Determines how long ago the user's most recent elite year was relative
        to when they wrote the review. Returns 100 if never elite.

        Args:
            user_elite: Comma-separated string of elite years
            review_date: Date when review was created

        Returns:
            Years since most recent elite year, or 100 if never elite

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> elite_str = "2010,2012,2015"
            >>> review_date = datetime(2018, 6, 1)
            >>> calc.years_since_elite_td(elite_str, review_date)
            # Returns 3 (2018 - 2015 = 3 years)
        """
        if user_elite in ["None", None, ""]:
            return 100

        try:
            elite_years = list(map(int, user_elite.split(",")))
            elite_before_review = [
                year for year in elite_years if year <= review_date.year
            ]

            if len(elite_before_review) == 0:
                return 100

            most_recent = max(elite_before_review)
            return review_date.year - most_recent
        except (ValueError, AttributeError):
            return 100

    @staticmethod
    def usefulness_level(ufc_count: Union[int, float]) -> str:
        """
        Categorize usefulness count into discrete levels.

        Creates categorical bins for vote counts, useful for multi-class
        classification tasks.

        Args:
            ufc_count: Total vote count (useful + funny + cool)

        Returns:
            Category string: "zero", "low", "medium", "high", or "unknown"

        Example:
            >>> TimeDiscountCalculator.usefulness_level(0)
            'zero'
            >>> TimeDiscountCalculator.usefulness_level(5)
            'medium'
            >>> TimeDiscountCalculator.usefulness_level(15)
            'high'
        """
        if ufc_count == 0:
            return "zero"
        elif ufc_count < 3:
            return "low"
        elif ufc_count < 10:
            return "medium"
        elif ufc_count >= 10:
            return "high"
        else:
            return "unknown"

    def create_all_targets(
        self,
        useful: Union[int, float],
        funny: Union[int, float],
        cool: Union[int, float],
        review_date: datetime,
        business_review_count: Union[int, float],
    ) -> dict:
        """
        Create all 6 target variable variations.

        Generates the complete set of target variables used in model training:
        - T1: Total votes (no discounting)
        - T2: Binary classification (has votes or not)
        - T3: Multi-class usefulness level (no discounting)
        - T4: Time-discounted votes (regression)
        - T5: Multi-class usefulness level (time-discounted)
        - T6: Time + business popularity discounted

        Args:
            useful: Useful vote count
            funny: Funny vote count
            cool: Cool vote count
            review_date: Date when review was created
            business_review_count: Number of reviews for the business

        Returns:
            Dictionary with all 6 target variables

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> targets = calc.create_all_targets(
            ...     useful=10, funny=5, cool=3,
            ...     review_date=datetime(2018, 6, 1),
            ...     business_review_count=500
            ... )
            >>> targets.keys()
            dict_keys(['T1_REG_review_total_ufc', 'T2_CLS_ufc_>0', ...])
        """
        # T1: Raw total (no discounting)
        t1_total = useful + funny + cool

        # T2: Binary classification
        t2_bool = t1_total > 0

        # T3: Categorical level (no discounting)
        t3_level = self.usefulness_level(t1_total)

        # T4: Time discounted
        t4_td = self.target_time_discount(t1_total, review_date)

        # T5: Categorical level (time discounted)
        t5_level = self.usefulness_level(t4_td)

        # T6: Time + business popularity discounted
        if business_review_count > 0:
            t6_tdbd = t4_td / business_review_count
        else:
            t6_tdbd = t4_td

        return {
            "T1_REG_review_total_ufc": t1_total,
            "T2_CLS_ufc_>0": t2_bool,
            "T3_CLS_ufc_level": t3_level,
            "T4_REG_ufc_TD": t4_td,
            "T5_CLS_ufc_level_TD": t5_level,
            "T6_REG_ufc_TDBD": t6_tdbd,
        }

    def discount_user_features(
        self,
        user_yelping_since: datetime,
        review_date: datetime,
        user_total_ufc: Union[int, float] = 0,
        user_compliments: Union[int, float] = 0,
        user_review_count: Union[int, float] = 0,
        user_fans: Union[int, float] = 0,
        user_friend_count: Union[int, float] = 0,
    ) -> dict:
        """
        Apply time discounting to all user count features.

        Creates time-discounted versions of all user features that accumulate
        over time (reviews, fans, friends, votes, compliments).

        Args:
            user_yelping_since: Date user joined Yelp
            review_date: Date when review was created
            user_total_ufc: Total user votes (useful + funny + cool) at dataset release
            user_compliments: Total compliments at dataset release
            user_review_count: Total reviews at dataset release
            user_fans: Total fans at dataset release
            user_friend_count: Total friends at dataset release

        Returns:
            Dictionary with time-discounted features (suffix: _TD)

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> discounted = calc.discount_user_features(
            ...     user_yelping_since=datetime(2010, 1, 1),
            ...     review_date=datetime(2015, 6, 1),
            ...     user_review_count=200,
            ...     user_fans=50
            ... )
            >>> discounted['user_review_count_TD']  # ~100 (halfway through tenure)
        """
        features = {
            "user_total_ufc": user_total_ufc,
            "user_compliments": user_compliments,
            "user_review_count": user_review_count,
            "user_fans": user_fans,
            "user_friend_count": user_friend_count,
        }

        discounted = {}
        for feature_name, feature_value in features.items():
            discounted[f"{feature_name}_TD"] = self.user_time_discount(
                feature_value, user_yelping_since, review_date
            )

        return discounted

    def discount_business_features(
        self,
        review_date: datetime,
        business_review_count: Union[int, float] = 0,
        business_checkin_count: Union[int, float] = 0,
    ) -> dict:
        """
        Apply time discounting to all business count features.

        Creates time-discounted versions of business features that accumulate
        over time (reviews, checkins).

        Args:
            review_date: Date when review was created
            business_review_count: Total business reviews at dataset release
            business_checkin_count: Total checkins at dataset release

        Returns:
            Dictionary with time-discounted features (suffix: _TD)

        Example:
            >>> calc = TimeDiscountCalculator()
            >>> discounted = calc.discount_business_features(
            ...     review_date=datetime(2015, 1, 1),
            ...     business_review_count=500,
            ...     business_checkin_count=1000
            ... )
            >>> discounted['business_review_count_TD']
        """
        features = {
            "business_review_count": business_review_count,
            "business_checkin_count": business_checkin_count,
        }

        discounted = {}
        for feature_name, feature_value in features.items():
            discounted[f"{feature_name}_TD"] = self.business_time_discount(
                feature_value, review_date
            )

        return discounted


# Convenience function for quick imports
def get_calculator() -> TimeDiscountCalculator:
    """
    Get a TimeDiscountCalculator instance with default settings.

    Returns:
        TimeDiscountCalculator with default dataset and founding dates
    """
    return TimeDiscountCalculator()


if __name__ == "__main__":
    # Example usage and testing
    calc = TimeDiscountCalculator()

    # Test target discounting
    review_dt = datetime(2018, 6, 1)
    votes = 42
    discounted = calc.target_time_discount(votes, review_dt)
    print(f"Target time discount: {votes} votes → {discounted:.2f}")

    # Test all targets
    targets = calc.create_all_targets(
        useful=10, funny=5, cool=3, review_date=review_dt, business_review_count=500
    )
    print(f"\nAll targets: {targets}")

    # Test user discounting
    user_since = datetime(2010, 1, 1)
    user_discounted = calc.discount_user_features(
        user_yelping_since=user_since,
        review_date=review_dt,
        user_review_count=200,
        user_fans=50,
    )
    print(f"\nUser features discounted: {user_discounted}")

    # Test business discounting
    biz_discounted = calc.discount_business_features(
        review_date=review_dt, business_review_count=500, business_checkin_count=1000
    )
    print(f"\nBusiness features discounted: {biz_discounted}")

    # Test elite functions
    elite_str = "2010,2012,2015,2018"
    elite_count = calc.count_elite_td(elite_str, review_dt)
    years_since = calc.years_since_elite_td(elite_str, review_dt)
    print(f"\nElite: {elite_count} years, {years_since} years since most recent")
