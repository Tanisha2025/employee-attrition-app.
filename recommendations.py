"""Rule-based HR recommendation engine. Priority: 1 = act now, 2 = soon, 3 = good practice."""

RATING = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}


def get_recommendations(row: dict, peer_income: dict) -> list:
    recs = []

    def add(priority, area, issue, action):
        recs.append({"priority": priority, "area": area,
                     "issue": issue, "action": action})

    # --- Workload & balance
    if row["OverTime"] == "Yes":
        add(1, "Workload", "Regularly works overtime",
            "Review workload and redistribute tasks; offer comp-off or "
            "overtime pay and set a cap on extra hours.")
    if row["WorkLifeBalance"] <= 2:
        add(1, "Work-life balance",
            f"Work-life balance rated {RATING[row['WorkLifeBalance']].lower()}",
            "Offer flexible hours or hybrid/remote days and encourage use of leave.")
    if row["BusinessTravel"] == "Travel_Frequently":
        add(2, "Travel", "Travels frequently",
            "Rotate travel across the team, cap travel days per month and "
            "offer travel-recovery days.")
    if row["DistanceFromHome"] >= 15:
        add(2, "Commute", f"Lives {row['DistanceFromHome']} km from work",
            "Offer hybrid/remote days, shuttle or a commute allowance.")

    # --- Engagement
    if row["JobSatisfaction"] <= 2:
        add(1, "Engagement",
            f"Job satisfaction rated {RATING[row['JobSatisfaction']].lower()}",
            "Hold a one-to-one stay interview this week; explore role "
            "enrichment, new projects or a change of team.")
    if row["EnvironmentSatisfaction"] <= 2:
        add(2, "Environment",
            f"Environment satisfaction rated {RATING[row['EnvironmentSatisfaction']].lower()}",
            "Run a short pulse survey with the team and fix the top two "
            "complaints (space, tools, team culture).")
    if row["JobInvolvement"] <= 2:
        add(2, "Engagement", "Low job involvement",
            "Give ownership of a visible project or a mentoring role.")

    # --- Career growth
    if row["YearsSinceLastPromotion"] >= 4:
        add(1, "Career growth",
            f"No promotion for {row['YearsSinceLastPromotion']} years",
            "Review promotion readiness and agree a written career path "
            "with a timeline.")
    if row["TrainingTimesLastYear"] <= 1:
        add(3, "Learning", "Little training last year",
            "Enrol in a skills programme or certification linked to the career path.")
    if row["NumCompaniesWorked"] >= 5 and row["YearsAtCompany"] <= 3:
        add(2, "Retention", "History of frequent job changes",
            "Assign a mentor and set 6- and 12-month goals to build commitment early.")

    # --- Pay & rewards
    median_pay = peer_income.get(int(row["JobLevel"]))
    if median_pay and row["MonthlyIncome"] < 0.85 * median_pay:
        add(1, "Compensation",
            f"Pay is {100 * (1 - row['MonthlyIncome'] / median_pay):.0f}% "
            f"below the median for job level {int(row['JobLevel'])}",
            "Run a compensation benchmark and consider a market adjustment or retention bonus.")
    if row["StockOptionLevel"] == 0:
        add(3, "Rewards", "No stock options",
            "Check eligibility for ESOP/long-term incentives.")

    # --- Manager & onboarding
    if row["YearsAtCompany"] <= 2:
        add(2, "Onboarding", "New joiner (2 years or less)",
            "Pair with a buddy and run 30-60-90 day check-ins.")
    if row["YearsWithCurrManager"] <= 1 and row["YearsAtCompany"] >= 3:
        add(3, "Manager", "Recently changed manager",
            "Schedule fortnightly manager check-ins for the next quarter.")

    if not recs:
        add(3, "General", "No major risk factors found",
            "Keep up regular check-ins and recognise good work.")
    return sorted(recs, key=lambda r: r["priority"])
