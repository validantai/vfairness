"""
A. Historical Pattern Detection

Detects bias patterns rooted in historical discrimination by cross-referencing
dataset features with known discriminatory patterns and practices.

Key capabilities:
- Identify features linked to historical discrimination (redlining, segregation, etc.)
- Detect temporal bias patterns in time-series data
- Cross-reference geographic features with historical discrimination maps
- Flag features that encode historical inequities

Academic References:
    - Barocas, S. & Selbst, A.D. (2016). Big Data's Disparate Impact.
      California Law Review, 104(3), 671-732.
      https://doi.org/10.15779/Z38BG31

    - Eubanks, V. (2018). Automating Inequality: How High-Tech Tools Profile,
      Police, and Punish the Poor. St. Martin's Press.
      ISBN: 978-1250074317

    - Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).
      Dissecting racial bias in an algorithm used to manage the health of populations.
      Science, 366(6464), 447-453.
      https://doi.org/10.1126/science.aax2342

    - Bertrand, M. & Mullainathan, S. (2004). Are Emily and Greg More Employable
      than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination.
      American Economic Review, 94(4), 991-1013.
      https://doi.org/10.1257/0002828042002561

    - Rothstein, R. (2017). The Color of Law: A Forgotten History of How Our
      Government Segregated America. Liveright Publishing.
      ISBN: 978-1631492853

Official Data Sources & Databases:
    - HOLC Redlining Maps: University of Richmond's Mapping Inequality Project
      https://dsl.richmond.edu/panorama/redlining/

    - EEOC Guidelines on Criminal Records:
      https://www.eeoc.gov/laws/guidance/enforcement-guidance-consideration-arrest-and-conviction-records

    - FFIEC Home Mortgage Disclosure Act (HMDA) Data:
      https://ffiec.cfpb.gov/

    - US Census Bureau Demographic Data:
      https://data.census.gov/

    - CFPB Consumer Complaint Database:
      https://www.consumerfinance.gov/data-research/consumer-complaints/

    - HUD Fair Housing & Equal Opportunity:
      https://www.hud.gov/program_offices/fair_housing_equal_opp

    - CDC Social Vulnerability Index (SVI):
      https://www.atsdr.cdc.gov/placeandhealth/svi/

Note:
    The patterns in this module are based on academic research and documented
    historical practices. For actual integration with official databases (e.g.,
    HOLC maps, HMDA data), see the `geographic_data` module.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from vfairness._names import name_token_list, tokens_contain
from vfairness._not_assessed import NOT_ASSESSED


class HistoricalRiskLevel(Enum):
    """Risk levels for historical bias patterns."""

    CRITICAL = "critical"  # Direct encoding of historical discrimination
    HIGH = "high"  # Strong correlation with historical patterns
    MEDIUM = "medium"  # Moderate correlation, requires investigation
    LOW = "low"  # Weak signals, monitor
    NONE = "none"  # No detected historical pattern


@dataclass
class HistoricalPatternResult:
    """
    Result of historical pattern detection for a single feature.

    Attributes:
        feature: Column name analyzed
        risk_level: Assessed risk level for historical bias
        pattern_type: Type of historical pattern detected
        description: Human-readable description of the finding
        historical_context: Relevant historical context explaining the risk
        affected_groups: Groups potentially affected by this pattern
        confidence: Confidence score (0-1) in the detection
        recommendations: Suggested actions to address the risk
        evidence: Supporting evidence for the detection. May carry
            ``protected_correlations`` (measured) and
            ``protected_correlations_not_assessable`` (attribute -> why no
            correlation could be taken); the second is never a 0.0 in the first.
    """

    feature: str
    risk_level: HistoricalRiskLevel
    pattern_type: str
    description: str
    historical_context: str
    affected_groups: List[str]
    confidence: float
    recommendations: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "risk_level": self.risk_level.value,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "historical_context": self.historical_context,
            "affected_groups": self.affected_groups,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "evidence": self.evidence,
        }


# HISTORICAL RISK PATTERNS CATALOG
# Comprehensive catalog of features and patterns linked to historical discrimination

HISTORICAL_RISK_PATTERNS: Dict[str, Dict[str, Any]] = {
    # GEOGRAPHIC DISCRIMINATION (Redlining, Segregation)
    "redlining_geographic": {
        "keywords": [
            "zip",
            "zipcode",
            "zip_code",
            "postal",
            "postcode",
            "postal_code",
            "neighborhood",
            "neighbourhood",
            "census_tract",
            "block_group",
            "census_block",
            "ward_code",
            "electoral_ward",
            "school_district",
            "district_code",
            "fips",
            "geoid",
            "tract",
            "block",
            "district",
            "ward",
        ],
        "pattern_type": "Geographic Redlining",
        "historical_context": (
            "Redlining was a discriminatory practice where banks and insurers "
            "refused services to residents of certain neighborhoods, predominantly "
            "those with Black and minority populations. The Home Owners' Loan "
            "Corporation (HOLC) maps from the 1930s-1940s created lasting segregation "
            "patterns that persist today. ZIP codes and census tracts often encode "
            "this historical discrimination."
        ),
        "affected_groups": ["Black/African American", "Hispanic/Latino", "Low-income"],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Evaluate if geographic feature is necessary for the model's purpose",
            "Test for disparate impact by cross-referencing with demographic data",
            "Consider using broader geographic aggregations to reduce discrimination risk",
            "Implement fairness constraints if geographic features must be used",
        ],
    },
    "neighborhood_names": {
        "keywords": [
            "area_name",
            "neighborhood_name",
            "district_name",
            "community",
            "locality",
            "suburb",
            "borough",
            "quarter",
        ],
        "pattern_type": "Neighborhood-based Discrimination",
        "historical_context": (
            "Neighborhood names can encode historical segregation patterns. "
            "Certain neighborhoods were designated for specific racial or ethnic "
            "groups through restrictive covenants, zoning laws, and informal practices. "
            "Names of historically segregated neighborhoods may trigger biased decisions."
        ),
        "affected_groups": ["Racial minorities", "Ethnic minorities", "Immigrants"],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Avoid using neighborhood names as direct features",
            "If needed, aggregate to larger geographic units",
            "Test for correlation with protected attributes",
        ],
    },
    # EDUCATIONAL DISCRIMINATION
    "educational_institution": {
        "keywords": [
            "school",
            "school_name",
            "university",
            "college",
            "alma_mater",
            "institution",
            "education_institution",
            "school_district",
            "school_code",
            "school_id",
        ],
        "pattern_type": "Educational Institutional Bias",
        "historical_context": (
            "Educational institutions have historically been segregated and unequally "
            "funded. School names and districts often reflect historical patterns of "
            "segregation and resource inequality. Using specific institution names can "
            "perpetuate advantages for historically privileged groups and disadvantages "
            "for those from underfunded schools."
        ),
        "affected_groups": ["Racial minorities", "Low-income", "First-generation students"],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Focus on skills and qualifications rather than institution names",
            "If using education features, use degree level rather than institution",
            "Consider blind review processes that remove institution names",
        ],
    },
    "legacy_admissions": {
        "keywords": [
            "legacy",
            "legacy_status",
            "parent_alumni",
            "family_connection",
            "donor",
            "donor_status",
            "development_case",
        ],
        "pattern_type": "Legacy Preference",
        "historical_context": (
            "Legacy preferences in education perpetuate historical advantages "
            "for families who could attend elite institutions when they excluded "
            "minorities. This creates a compounding advantage for historically "
            "privileged groups across generations."
        ),
        "affected_groups": ["First-generation students", "Racial minorities", "Low-income"],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Exclude legacy status from predictive models",
            "If used for analysis, control for socioeconomic status",
            "Consider its disparate impact on underrepresented groups",
        ],
    },
    # EMPLOYMENT DISCRIMINATION
    "employment_gaps": {
        "keywords": [
            "gap",
            "employment_gap",
            "career_gap",
            "resume_gap",
            "work_gap",
            "time_since_employment",
            "months_unemployed",
            "years_out",
        ],
        "pattern_type": "Employment Gap Penalty",
        "historical_context": (
            "Employment gaps disproportionately affect women (due to caregiving), "
            "individuals with disabilities or health conditions, formerly incarcerated "
            "individuals, and those affected by economic downturns that hit minority "
            "communities harder. Penalizing gaps perpetuates these historical inequities."
        ),
        "affected_groups": ["Women", "Caregivers", "Disabled individuals", "Formerly incarcerated"],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Evaluate skills and qualifications rather than continuous employment",
            "Allow candidates to explain gaps",
            "Consider the context of gaps (e.g., COVID-19, caregiving)",
        ],
    },
    "salary_history": {
        "keywords": [
            "salary_history",
            "previous_salary",
            "current_salary",
            "last_salary",
            "compensation_history",
            "pay_history",
            "prior_earnings",
        ],
        "pattern_type": "Salary History Perpetuation",
        "historical_context": (
            "Using salary history perpetuates the gender and racial wage gaps. "
            "Women and minorities have historically been paid less for the same work, "
            "and basing new offers on previous pay locks in this discrimination. "
            "Many jurisdictions have banned salary history inquiries for this reason."
        ),
        "affected_groups": ["Women", "Racial minorities", "Young workers"],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Do not use salary history in hiring or compensation decisions",
            "Base offers on the role's value and candidate qualifications",
            "Check if salary history bans apply in your jurisdiction",
        ],
    },
    # CRIMINAL JUSTICE DISCRIMINATION
    "criminal_records": {
        "keywords": [
            "arrest",
            "conviction",
            "criminal",
            "felony",
            "misdemeanor",
            "incarceration",
            "prison",
            "jail",
            "probation",
            "parole",
            "background_check",
            "criminal_history",
            "offense",
            "charge",
        ],
        "pattern_type": "Criminal Justice Bias",
        "historical_context": (
            "The criminal justice system has a well-documented history of racial "
            "bias in policing, prosecution, and sentencing. Black Americans are "
            "incarcerated at 5x the rate of white Americans. Using criminal records "
            "perpetuates this bias, as arrest records reflect policing patterns, "
            "not just criminal behavior."
        ),
        "affected_groups": ["Black/African American", "Hispanic/Latino", "Low-income"],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Distinguish between arrests and convictions (arrests reflect policing bias)",
            "Consider relevance of offense to the decision being made",
            "Apply individualized assessment rather than blanket exclusions",
            "Follow EEOC guidance on criminal record usage",
            "Recognize expunged records should not be used",
        ],
    },
    # HEALTHCARE DISCRIMINATION
    "healthcare_cost": {
        "keywords": [
            "healthcare_cost",
            "medical_cost",
            "health_expenditure",
            "claims_cost",
            "treatment_cost",
            "medical_spending",
            "healthcare_utilization",
            "health_spending",
        ],
        "pattern_type": "Healthcare Cost as Health Proxy",
        "historical_context": (
            "Obermeyer et al. (2019) demonstrated that using healthcare costs as a "
            "proxy for health needs systematically disadvantages Black patients. "
            "Due to historical barriers to healthcare access, Black patients incur "
            "lower costs at the same level of illness, leading algorithms to "
            "underestimate their health needs."
        ),
        "affected_groups": ["Black/African American", "Low-income", "Uninsured/Underinsured"],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Use direct health measures rather than cost as a proxy for health needs",
            "Audit algorithms for racial disparities in predictions",
            "Consider access barriers when interpreting healthcare data",
        ],
    },
    # FINANCIAL DISCRIMINATION
    "credit_history": {
        "keywords": [
            "credit_score",
            "credit_history",
            "fico",
            "credit_rating",
            "credit_report",
            "payment_history",
            "delinquency",
            "default_history",
            "bankruptcy",
        ],
        "pattern_type": "Credit History Bias",
        "historical_context": (
            "Credit scoring systems encode historical discrimination in lending. "
            "Redlining, predatory lending, and lack of access to mainstream banking "
            "have left minority communities with lower credit scores on average. "
            "Using credit scores can perpetuate these historical inequities."
        ),
        "affected_groups": [
            "Black/African American",
            "Hispanic/Latino",
            "Low-income",
            "Young adults",
        ],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Consider whether credit history is truly predictive for your use case",
            "Use alternative data sources that don't encode historical discrimination",
            "Test for disparate impact across demographic groups",
        ],
    },
    "banking_access": {
        "keywords": [
            "bank_account",
            "banking_status",
            "unbanked",
            "underbanked",
            "checking_account",
            "savings_account",
            "account_tenure",
        ],
        "pattern_type": "Banking Access Discrimination",
        "historical_context": (
            "Access to banking has historically been unequal. Minority and low-income "
            "communities are more likely to be unbanked or underbanked due to bank "
            "branch closures in their neighborhoods, discriminatory practices, and "
            "historical exclusion from the financial system."
        ),
        "affected_groups": ["Racial minorities", "Low-income", "Rural communities", "Immigrants"],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Avoid penalizing lack of traditional banking relationships",
            "Consider alternative financial behaviors as indicators",
            "Recognize that banking deserts affect certain communities disproportionately",
        ],
    },
    # NAME-BASED DISCRIMINATION
    "names": {
        "keywords": [
            "name",
            "first_name",
            "last_name",
            "surname",
            "given_name",
            "family_name",
            "full_name",
            "applicant_name",
            "candidate_name",
        ],
        "pattern_type": "Name-based Discrimination",
        "historical_context": (
            "Research shows significant discrimination based on names that signal "
            "race, ethnicity, gender, or religion. The Bertrand & Mullainathan (2004) "
            "study found that resumes with 'white-sounding' names received 50% more "
            "callbacks than identical resumes with 'Black-sounding' names."
        ),
        "affected_groups": [
            "Racial minorities",
            "Ethnic minorities",
            "Women",
            "Religious minorities",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Remove names from decision-making processes where possible",
            "Implement blind review for hiring and admissions",
            "Never use name-based predictions of demographic attributes",
        ],
    },
    # TECHNOLOGY ACCESS DISCRIMINATION
    "digital_access": {
        "keywords": [
            "device_type",
            "browser",
            "operating_system",
            "internet_speed",
            "connection_type",
            "mobile_vs_desktop",
            "app_version",
        ],
        "pattern_type": "Digital Divide",
        "historical_context": (
            "The digital divide reflects socioeconomic inequalities. Low-income and "
            "rural communities have less access to high-speed internet and modern "
            "devices. Using technology access signals can discriminate against "
            "these populations, creating a new form of digital redlining."
        ),
        "affected_groups": ["Low-income", "Rural communities", "Elderly", "Disabled individuals"],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Ensure services are accessible across device types and connection speeds",
            "Do not use device/browser signals for credit or risk decisions",
            "Test for disparate impact of technology requirements",
        ],
    },
    # EUROPEAN AI FAIRNESS PATTERNS
    # MIGRATION & NATIONALITY DISCRIMINATION (EU)
    "migration_background": {
        "keywords": [
            "nationality",
            "migration",
            "migrationshintergrund",
            "migration_background",
            "country_of_origin",
            "country_of_birth",
            "birth_country",
            "birthplace",
            "citizenship",
            "dual_nationality",
            "dual_citizenship",
            "residence_permit",
            "immigration_status",
            "visa_status",
            "refugee",
            "asylum",
            "migrant",
            "foreign_born",
            "native_born",
            "second_generation",
            "ethnic_origin",
        ],
        "pattern_type": "Migration Background Discrimination",
        "historical_context": (
            "In Europe, migration background (Migrationshintergrund in German) is a "
            "common proxy for ethnicity and race. The Dutch Childcare Benefits Scandal "
            "(Toeslagenaffaire, 2020) demonstrated how algorithms targeting families "
            "with dual nationality for fraud led to the wrongful persecution of 26,000 "
            "families and the resignation of the Dutch government. Nationality and "
            "country-of-birth features encode colonial histories, guest-worker programs "
            "(Gastarbeiter), and refugee patterns across Europe."
        ),
        "affected_groups": [
            "Immigrants",
            "Dual nationals",
            "Refugees",
            "Ethnic minorities",
            "Post-colonial communities",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Never use nationality or dual citizenship as fraud indicators",
            "Audit algorithms for disparate impact on migrant communities",
            "Comply with EU Race Equality Directive 2000/43/EC",
            "Consider that migration background correlates with socioeconomic disadvantage",
            "Reference: Dutch Toeslagenaffaire Parliamentary Inquiry (2020)",
        ],
    },
    # EUROPEAN WELFARE & BENEFITS ALGORITHMS
    "welfare_fraud_scoring": {
        "keywords": [
            "fraud_score",
            "fraud_risk",
            "fraud_probability",
            "risk_score",
            "risk_profile",
            "benefit_fraud",
            "welfare_fraud",
            "overpayment_risk",
            "compliance_score",
            "integrity_score",
            "toeslagen",
            "benefits",
            "social_welfare",
            "social_security",
            "unemployment_benefit",
            "housing_benefit",
            "child_benefit",
            "allowance",
        ],
        "pattern_type": "Welfare Fraud Algorithm Bias",
        "historical_context": (
            "European welfare fraud detection systems have shown systematic bias "
            "against ethnic minorities and low-income communities. The Dutch SyRI "
            "(System Risk Indication) was ruled a violation of ECHR Article 8 by "
            "The Hague District Court (2020, NJCM v. Netherlands). The system "
            "disproportionately targeted disadvantaged neighbourhoods. Similarly, "
            "the UK's Universal Credit algorithm and DWP fraud detection have been "
            "criticised for discriminatory profiling."
        ),
        "affected_groups": [
            "Ethnic minorities",
            "Low-income",
            "Single parents",
            "Disabled individuals",
            "Immigrants",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Audit fraud detection for disparate impact across protected groups",
            "Do not target specific neighbourhoods or demographics for enhanced scrutiny",
            "Ensure human oversight per EU AI Act requirements for high-risk systems",
            "Comply with ECHR Article 8 (right to private life) and GDPR Article 22",
            "Reference: The Hague District Court, SyRI ruling (2020)",
        ],
    },
    # EUROPEAN CREDIT & FINANCIAL SCORING
    "european_credit_scoring": {
        "keywords": [
            "schufa",
            "schufa_score",
            "kredit_score",
            "bonität",
            "bonitaet",
            "credit_bureau",
            "experian",
            "equifax",
            "creditreform",
            "postcode_score",
            "plz_score",
            "geo_score",
            "area_score",
            "neighbourhood_score",
            "neighborhood_score",
            "address_score",
        ],
        "pattern_type": "European Credit Scoring Bias",
        "historical_context": (
            "European credit scoring systems like SCHUFA (Germany), Experian (UK), "
            "and similar bureaus use opaque algorithms that can encode geographic and "
            "socioeconomic discrimination. The German Federal Court (BGH, 2024) ruled "
            "on SCHUFA transparency obligations. Postcode-based scoring "
            "(Postleitzahlen-Scoring) is the European equivalent of redlining, using "
            "neighbourhood demographics to assess individual creditworthiness. GDPR "
            "Article 22 gives individuals the right not to be subject to purely "
            "automated decisions with significant effects."
        ),
        "affected_groups": [
            "Immigrants",
            "Low-income",
            "Young adults",
            "Residents of disadvantaged areas",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Ensure credit scoring complies with GDPR Article 22 (automated decisions)",
            "Avoid postcode/PLZ-based scoring that proxies neighbourhood demographics",
            "Provide meaningful explanations of credit decisions per GDPR Article 15",
            "Test for disparate impact across ethnic and socioeconomic groups",
            "Reference: BGH SCHUFA ruling (2024); CJEU C-634/21 (2023)",
        ],
    },
    # EDUCATIONAL ALGORITHMS (EU)
    "exam_grading_algorithms": {
        "keywords": [
            "predicted_grade",
            "estimated_grade",
            "grade_prediction",
            "school_type",
            "school_category",
            "school_ranking",
            "school_performance",
            "centre_assessment",
            "teacher_assessment",
            "historical_results",
            "school_history",
            "school_deprivation",
            "free_school_meals",
            "fsm",
            "pupil_premium",
        ],
        "pattern_type": "Educational Grading Algorithm Bias",
        "historical_context": (
            "The 2020 UK A-Level/GCSE algorithm scandal (Ofqual) demonstrated how "
            "standardisation algorithms can systematically disadvantage students from "
            "state schools and deprived areas. The algorithm used school historical "
            "performance to moderate teacher-assessed grades, upgrading private school "
            "students while downgrading state school students, disproportionately "
            "affecting ethnic minority and low-income pupils. The policy was reversed "
            "after widespread protests."
        ),
        "affected_groups": [
            "Students from deprived areas",
            "Ethnic minority students",
            "State school students",
            "First-generation university applicants",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Do not use school-level historical performance to penalise individual students",
            "Audit grading algorithms for disparate impact by school type and demographics",
            "Ensure appeals processes are accessible to disadvantaged students",
            "Reference: Ofqual Review (2020); UK Education Select Committee Report",
        ],
    },
    # EMPLOYMENT ALGORITHMS (EU)
    "employment_profiling_eu": {
        "keywords": [
            "employability_score",
            "employability",
            "job_readiness",
            "ams_score",
            "labour_market_score",
            "reintegration_score",
            "jobseeker_profile",
            "activation_score",
            "workfare",
            "disability_status",
            "work_capacity",
            "reduced_capacity",
            "occupational_disability",
            "erwerbsminderung",
        ],
        "pattern_type": "Employment Profiling Algorithm Bias",
        "historical_context": (
            "The Austrian AMS (Arbeitsmarktservice) algorithm scored jobseekers to "
            "allocate training resources. Women received systematically lower scores "
            "due to statistical patterns of career interruptions (caregiving), and "
            "disabled individuals were scored lower due to perceived labour market "
            "barriers. The algorithm effectively denied resources to those who needed "
            "them most. The Austrian Data Protection Authority intervened, and "
            "AlgorithmWatch documented the case extensively."
        ),
        "affected_groups": [
            "Women",
            "Disabled individuals",
            "Older workers",
            "Long-term unemployed",
            "Migrants",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Do not use gender or disability as inputs to employment scoring",
            "Audit for indirect discrimination through correlated features",
            "Ensure human oversight for resource allocation decisions",
            "Comply with EU Employment Equality Directive 2000/78/EC",
            "Reference: Austrian DPA ruling on AMS; AlgorithmWatch (2019)",
        ],
    },
    # PREDICTIVE POLICING (EU)
    "predictive_policing_eu": {
        "keywords": [
            "crime_risk",
            "crime_score",
            "offender_score",
            "reoffending",
            "recidivism",
            "gang_score",
            "gangs_matrix",
            "hotspot",
            "crime_hotspot",
            "police_risk",
            "threat_score",
            "cas_score",
            "precobs",
            "predpol",
            "crime_prediction",
        ],
        "pattern_type": "Predictive Policing Bias",
        "historical_context": (
            "European predictive policing systems have been shown to encode and "
            "amplify ethnic profiling. The UK Metropolitan Police Gangs Matrix was "
            "found by Amnesty International (2018) to disproportionately target "
            "Black individuals: 78%% of entries were Black in a city where they "
            "are 13%% of the population. The Dutch CAS (Crime Anticipation System) "
            "and German PRECOBS have raised similar concerns. These systems create "
            "feedback loops: over-policed communities generate more crime data, "
            "leading to more policing."
        ),
        "affected_groups": [
            "Black communities",
            "Ethnic minorities",
            "Roma",
            "Young men",
            "Low-income neighbourhoods",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Audit for racial and ethnic disproportionality in risk scores",
            "Recognise that crime data reflects policing patterns, not just crime",
            "Ensure compliance with EU AI Act prohibition on social scoring",
            "Implement safeguards against feedback loops in crime prediction",
            "Reference: Amnesty International (2018); Dutch DPA ruling on CAS",
        ],
    },
    # ROMA & TRAVELLER DISCRIMINATION
    "roma_traveller_discrimination": {
        "keywords": [
            "roma",
            "traveller",
            "gypsy",
            "sinti",
            "nomad",
            "itinerant",
            "settled_status",
            "accommodation_type",
            "dwelling_type",
            "caravan",
            "halting_site",
            "encampment",
            "fixed_abode",
            "no_fixed_abode",
            "nfa",
        ],
        "pattern_type": "Roma and Traveller Discrimination",
        "historical_context": (
            "Roma and Traveller communities are Europe's most discriminated minority. "
            "The EU Fundamental Rights Agency (FRA) documents pervasive discrimination "
            "in housing, education, healthcare, and employment. Accommodation type "
            "and 'no fixed abode' features directly encode anti-Roma prejudice. "
            "Predictive systems in child protection and welfare have been shown to "
            "disproportionately flag Roma families. The Council of Europe's ECRI "
            "has issued multiple country reports on anti-Gypsyism in algorithmic systems."
        ),
        "affected_groups": ["Roma", "Sinti", "Travellers", "Nomadic communities"],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Never use accommodation type as a risk factor",
            "Audit child protection and welfare algorithms for anti-Roma bias",
            "Comply with EU Framework for Roma Equality (2020-2030)",
            "Ensure data collection does not stigmatise Roma communities",
            "Reference: FRA EU-MIDIS II (2016); Council of Europe ECRI reports",
        ],
    },
    # RELIGIOUS IDENTITY / APPEARANCE (EU)
    "religious_identity_eu": {
        "keywords": [
            "religion",
            "religious",
            "faith",
            "confession",
            "denomination",
            "headscarf",
            "hijab",
            "veil",
            "religious_symbol",
            "religious_dress",
            "church_tax",
            "kirchensteuer",
            "halal",
            "kosher",
            "religious_holiday",
            "sabbath",
            "prayer_time",
        ],
        "pattern_type": "Religious Identity Discrimination",
        "historical_context": (
            "Religious discrimination in Europe intersects with ethnic and migration "
            "discrimination, particularly for Muslim communities. CV studies across "
            "Europe show that applicants with Muslim-sounding names or headscarf photos "
            "receive significantly fewer callbacks. The ECHR has ruled on religious "
            "symbol cases (Dahlab v. Switzerland, 2001; Şahin v. Turkey, 2005). "
            "In Germany, the church tax (Kirchensteuer) field in employment data "
            "directly reveals religious affiliation. France's laïcité framework and "
            "Belgian headscarf bans create additional algorithmic risks."
        ),
        "affected_groups": [
            "Muslims",
            "Jewish communities",
            "Sikhs",
            "Religious minorities",
            "Women wearing religious dress",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Remove religious identifiers from decision-making features",
            "Audit name-based systems for anti-Muslim bias",
            "Do not use photo-based features that capture religious dress",
            "Comply with EU Charter of Fundamental Rights Article 21",
            "Reference: ECHR Dahlab (2001), Şahin (2005); EU Employment Directive",
        ],
    },
    # PLATFORM / GIG ECONOMY (EU)
    "platform_gig_scoring": {
        "keywords": [
            "rider_score",
            "driver_score",
            "worker_rating",
            "platform_rating",
            "customer_rating",
            "delivery_score",
            "performance_score",
            "gig_score",
            "freelancer_rating",
            "contractor_rating",
            "algorithmic_management",
            "task_allocation",
            "shift_allocation",
            "availability_score",
            "acceptance_rate",
            "cancellation_rate",
        ],
        "pattern_type": "Platform Work Algorithmic Bias",
        "historical_context": (
            "Algorithmic management in the gig economy has been shown to encode "
            "customer bias against ethnic minorities into worker ratings and task "
            "allocation. Italian courts ruled that Deliveroo's algorithm discriminated "
            "against workers who took time off for religious holidays or strikes. "
            "The EU Platform Workers Directive (2024) requires transparency in "
            "algorithmic management. Research shows that customer ratings reflect "
            "racial bias (lower ratings for non-white workers), which then affects "
            "job allocation, creating a discrimination feedback loop."
        ),
        "affected_groups": [
            "Ethnic minority workers",
            "Migrant workers",
            "Religious minorities",
            "Workers with disabilities",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Audit customer ratings for racial and ethnic bias before using in algorithms",
            "Do not penalise workers for religious holiday absences",
            "Comply with EU Platform Workers Directive transparency requirements",
            "Implement human review for deactivation and task allocation decisions",
            "Reference: Bologna Tribunal Deliveroo ruling (2021); EU Directive 2024/2831",
        ],
    },
    # BIOMETRIC & FACIAL RECOGNITION (EU)
    "biometric_identification": {
        "keywords": [
            "biometric",
            "facial_recognition",
            "face_id",
            "face_match",
            "face_score",
            "biometric_score",
            "fingerprint",
            "iris",
            "iris_scan",
            "iris_code",
            "iris_template",
            "liveness_score",
            "identity_verification",
            "kyc_score",
            "emotion_recognition",
            "sentiment_score",
            "facial_analysis",
            "age_estimation",
            "gender_detection",
        ],
        "pattern_type": "Biometric Identification Bias",
        "historical_context": (
            "Facial recognition and biometric systems have documented accuracy "
            "disparities across skin tones, gender, and age. Research by Buolamwini & "
            "Gebru (2018, Gender Shades) found error rates up to 34.7%% higher for "
            "dark-skinned women. The EU AI Act (2024) prohibits real-time remote "
            "biometric identification in public spaces (with limited exceptions) and "
            "classifies biometric categorisation as high-risk. Emotion recognition "
            "systems are banned in workplaces and education under the Act."
        ),
        "affected_groups": [
            "Dark-skinned individuals",
            "Women",
            "Elderly",
            "Trans and non-binary individuals",
            "People with disabilities",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Test biometric systems for accuracy disparities across demographics",
            "Comply with EU AI Act restrictions on biometric identification",
            "Never use emotion recognition in employment or education contexts",
            "Provide non-biometric alternatives for identity verification",
            "Reference: Buolamwini & Gebru (2018); EU AI Act (2024) Art. 5, Annex III",
        ],
    },
    # SOCIAL HOUSING & ACCOMMODATION (EU)
    "social_housing_eu": {
        "keywords": [
            "council_housing",
            "social_housing",
            "housing_allocation",
            "housing_priority",
            "housing_band",
            "waiting_list",
            "hlm",
            "sozialwohnung",
            "wohnungsvergabe",
            "housing_register",
            "homeless",
            "homelessness",
            "rough_sleeping",
            "temporary_accommodation",
            "housing_benefit",
            "wohngeld",
            "apl",
            "housing_association",
        ],
        "pattern_type": "Social Housing Algorithm Bias",
        "historical_context": (
            "Social housing allocation algorithms across Europe have been shown to "
            "encode neighbourhood segregation patterns. UK council housing allocation "
            "systems (choice-based lettings) can disadvantage non-English speakers "
            "and digitally excluded groups. French HLM (habitation à loyer modéré) "
            "allocation has been criticised for perpetuating ethnic segregation. "
            "Homelessness prediction models risk encoding systemic factors "
            "(racism, immigration status) as individual risk factors."
        ),
        "affected_groups": [
            "Ethnic minorities",
            "Immigrants",
            "Roma",
            "Large families",
            "Single parents",
            "Disabled individuals",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Audit housing allocation for disparate impact by ethnicity and nationality",
            "Ensure digital allocation systems are accessible in multiple languages",
            "Do not use neighbourhood-level deprivation to deny housing in that area",
            "Comply with EU Race Equality Directive in housing provision",
            "Reference: UK Shelter reports; French Défenseur des droits; FEANTSA",
        ],
    },
    # LANGUAGE & ACCENT DISCRIMINATION (EU)
    "language_discrimination": {
        "keywords": [
            "language",
            "native_language",
            "mother_tongue",
            "language_proficiency",
            "accent",
            "dialect",
            "language_test",
            "language_score",
            "multilingual",
            "monolingual",
            "interpreter",
            "translation",
            "language_barrier",
            "communication_score",
            "fluency",
            "minority_language",
            "regional_language",
        ],
        "pattern_type": "Language and Accent Discrimination",
        "historical_context": (
            "NLP and voice-based systems show bias against non-native speakers, "
            "regional dialects, and minority languages. Speech recognition accuracy "
            "drops significantly for accented speech, and automated interview systems "
            "can penalise non-native speakers. Europe's linguistic diversity (with "
            "minority languages like Welsh, Basque, Catalan, Romani, Sámi, and many "
            "others) means language-based features can proxy for ethnicity, "
            "nationality, and social class. The European Charter for Regional or "
            "Minority Languages protects linguistic diversity."
        ),
        "affected_groups": [
            "Non-native speakers",
            "Immigrants",
            "Regional minorities",
            "Deaf individuals",
            "Working class",
        ],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Test NLP/voice systems for accuracy across accents and dialects",
            "Do not use language proficiency as a proxy for competence in unrelated tasks",
            "Provide multilingual access to algorithmic decision systems",
            "Comply with European Charter for Regional or Minority Languages",
            "Reference: EU Charter Art. 21-22; Council of Europe language rights",
        ],
    },
    # EU AI ACT: PROHIBITED & HIGH-RISK PATTERNS (Regulation 2024/1689)
    # SOCIAL SCORING: PROHIBITED (Art. 5(1)(c))
    "euaia_social_scoring": {
        "keywords": [
            "social_score",
            "social_credit",
            "citizen_score",
            "civic_score",
            "trustworthiness_score",
            "reliability_score",
            "social_rating",
            "behaviour_score",
            "behavioral_score",
            "social_trustworthiness",
            "social_ranking",
            "citizen_ranking",
            "loyalty_score",
            "community_score",
            "societal_score",
            "social_compliance",
        ],
        "pattern_type": "Social Scoring: PROHIBITED under EU AI Act Art. 5(1)(c)",
        "historical_context": (
            "The EU AI Act (Regulation 2024/1689) Art. 5(1)(c) explicitly prohibits "
            "AI systems that evaluate or classify individuals based on their social "
            "behaviour or personal characteristics, where the resulting social score "
            "leads to detrimental or unfavourable treatment unrelated to the context "
            "in which the data was generated, or disproportionate to the social "
            "behaviour. This prohibition directly addresses China-style social credit "
            "systems but also covers any algorithmic social scoring that aggregates "
            "behavioural signals to rank individuals. The prohibition applies to both "
            "public authorities and private entities operating within the EU."
        ),
        "affected_groups": [
            "All individuals",
            "Ethnic minorities",
            "Low-income",
            "Politically active individuals",
            "Religious minorities",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "PROHIBITED: Remove any social scoring features (EU AI Act Art. 5(1)(c))",
            "Do not aggregate behavioural signals into general trustworthiness scores",
            "Ensure compliance: penalties up to €35 million or 7% of global turnover",
            "Audit existing scoring systems for social scoring characteristics",
            "Reference: EU AI Act Art. 5(1)(c); Recitals 31-32",
        ],
    },
    # EMOTION RECOGNITION: PROHIBITED in workplace/education (Art. 5(1)(f))
    "euaia_emotion_recognition": {
        "keywords": [
            "emotion_recognition",
            "emotion_detection",
            "emotion_score",
            "sentiment_analysis",
            "affect_recognition",
            "mood_detection",
            "facial_emotion",
            "voice_emotion",
            "stress_detection",
            "engagement_score",
            "attention_score",
            "fatigue_detection",
            "drowsiness_detection",
            "micro_expression",
            "emotional_state",
            "valence_arousal",
            "affective_computing",
            "emotion_ai",
        ],
        "pattern_type": "Emotion Recognition: PROHIBITED in workplace/education (Art. 5(1)(f))",
        "historical_context": (
            "The EU AI Act Art. 5(1)(f) prohibits AI systems that infer emotions "
            "of individuals in the workplace and educational institutions, except "
            "where intended for medical or safety purposes (e.g., driver drowsiness "
            "in safety-critical transport). The scientific basis for emotion "
            "recognition is contested: a 2019 meta-review by Barrett et al. in "
            "Psychological Science in the Public Interest found that facial "
            "expressions are not reliable indicators of emotional states. Systems "
            "show significant accuracy disparities by race and gender (Buolamwini "
            "& Gebru, 2018). The prohibition reflects both scientific uncertainty "
            "and the fundamental rights risks of workplace surveillance."
        ),
        "affected_groups": [
            "Workers",
            "Students",
            "Dark-skinned individuals",
            "Women",
            "Neurodiverse individuals",
            "People with disabilities",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "PROHIBITED in workplace and education: Remove emotion recognition features",
            "Only medical/safety exceptions apply (e.g., driver alertness monitoring)",
            "Do not use engagement or attention scores derived from facial/voice analysis",
            "Penalties: up to €35 million or 7% of global turnover",
            "Reference: EU AI Act Art. 5(1)(f); Barrett et al. (2019) PSiPI",
        ],
    },
    # BIOMETRIC CATEGORISATION by sensitive attributes: PROHIBITED (Art. 5(1)(g))
    "euaia_biometric_categorisation": {
        "keywords": [
            "biometric_category",
            "biometric_classification",
            "race_detection",
            "ethnicity_detection",
            "religion_detection",
            "sexual_orientation_detection",
            "political_opinion_detection",
            "skin_tone",
            "skin_color",
            "skin_colour",
            "race_prediction",
            "ethnicity_prediction",
            "gender_classification",
            "age_classification",
            "biometric_profiling",
            "facial_classification",
            "physiognomy",
            "phrenology",
        ],
        "pattern_type": "Biometric Categorisation: PROHIBITED (Art. 5(1)(g))",
        "historical_context": (
            "The EU AI Act Art. 5(1)(g) prohibits biometric categorisation systems "
            "that infer sensitive attributes including race, political opinions, trade "
            "union membership, religious beliefs, sex life, or sexual orientation. "
            "Exceptions exist only for labelling/filtering lawfully acquired biometric "
            "datasets and law enforcement categorisation. This prohibition addresses "
            "the pseudoscientific use of physiognomic features to predict behaviour "
            "or identity, which echoes discredited 19th-century practices. Research "
            "(Buolamwini & Gebru, 2018; Raji & Buolamwini, 2019) demonstrated "
            "systematic racial and gender bias in facial analysis systems."
        ),
        "affected_groups": [
            "Racial minorities",
            "Ethnic minorities",
            "Religious minorities",
            "LGBTQ+ individuals",
            "Political activists",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "PROHIBITED: Do not categorise individuals by sensitive attributes via biometrics",
            "Remove any race, ethnicity, or religion prediction features",
            "Exception only for lawful dataset labelling, not for individual decisions",
            "Penalties: up to €35 million or 7% of global turnover",
            "Reference: EU AI Act Art. 5(1)(g); Recital 44",
        ],
    },
    # SUBLIMINAL / MANIPULATIVE AI: PROHIBITED (Art. 5(1)(a))
    "euaia_manipulative_ai": {
        "keywords": [
            "dark_pattern",
            "dark_patterns",
            "nudge_score",
            "nudging",
            "subliminal",
            "manipulation_score",
            "persuasion_score",
            "addiction_score",
            "engagement_maximisation",
            "engagement_optimization",
            "attention_capture",
            "compulsive_usage",
            "retention_score",
            "behavioural_nudge",
            "behavioral_nudge",
            "choice_architecture",
            "deceptive_design",
            "addictive_design",
        ],
        "pattern_type": "Manipulative/Subliminal AI: PROHIBITED (Art. 5(1)(a))",
        "historical_context": (
            "The EU AI Act Art. 5(1)(a) prohibits AI systems deploying subliminal, "
            "manipulative, or deceptive techniques to materially distort behaviour "
            "in a manner that causes or is reasonably likely to cause significant harm. "
            "This covers dark patterns, addictive design, and AI-driven behavioural "
            "manipulation. Research by the European Commission's Joint Research Centre "
            "(JRC) and the OECD has documented how algorithmic choice architecture can "
            "exploit cognitive biases. The prohibition also covers AI systems that "
            "exploit vulnerabilities related to age, disability, or socio-economic "
            "circumstances (Art. 5(1)(b))."
        ),
        "affected_groups": [
            "Minors",
            "Elderly",
            "Persons with disabilities",
            "Low digital literacy individuals",
            "Economically vulnerable",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "PROHIBITED: Do not deploy subliminal or manipulative AI techniques",
            "Audit recommendation systems for manipulative engagement patterns",
            "Special protection required for vulnerable groups (minors, elderly, disabled)",
            "Penalties: up to €35 million or 7% of global turnover",
            "Reference: EU AI Act Art. 5(1)(a-b); Recitals 29-30",
        ],
    },
    # HIGH-RISK: RECRUITMENT & EMPLOYMENT (Annex III, area 4)
    "euaia_hr_recruitment": {
        "keywords": [
            "cv_score",
            "resume_score",
            "candidate_score",
            "candidate_ranking",
            "hiring_score",
            "recruitment_score",
            "interview_score",
            "applicant_ranking",
            "job_match_score",
            "talent_score",
            "screening_score",
            "shortlist_score",
            "fitment_score",
            "promotion_score",
            "termination_score",
            "performance_rating",
            "performance_review",
            "task_allocation_score",
            "workforce_analytics",
        ],
        "pattern_type": "High-Risk: Employment & Recruitment (Annex III, area 4)",
        "historical_context": (
            "The EU AI Act classifies AI systems used in recruitment, candidate "
            "evaluation, promotion, termination, task allocation based on personality "
            "traits, and performance monitoring as HIGH-RISK (Annex III, area 4). "
            "Providers must implement risk management, data governance, human oversight, "
            "and bias testing. Amazon's discontinued AI hiring tool (2018) demonstrated "
            "how training data can encode gender bias: it systematically downgraded "
            "women's CVs. The Austrian AMS algorithm (2019) similarly discriminated "
            "against women and disabled jobseekers. These systems require conformity "
            "assessment and registration in the EU database before deployment."
        ),
        "affected_groups": [
            "Women",
            "Ethnic minorities",
            "Older workers",
            "Persons with disabilities",
            "Career gap individuals",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Implement full conformity assessment",
            "Conduct bias testing across gender, ethnicity, age, disability before deployment",
            "Ensure human oversight: no fully automated hiring/termination decisions",
            "Register system in EU database; maintain technical documentation",
            "Reference: EU AI Act Annex III area 4; Art. 9-15",
        ],
    },
    # HIGH-RISK: CREDITWORTHINESS ASSESSMENT (Annex III, area 5(b))
    "euaia_creditworthiness": {
        "keywords": [
            "credit_decision",
            "creditworthiness",
            "credit_assessment",
            "loan_decision",
            "lending_decision",
            "credit_approval",
            "loan_approval",
            "mortgage_decision",
            "underwriting_decision",
            "credit_limit_decision",
            "default_prediction",
            "default_probability",
            "pd_score",
            "probability_of_default",
            "credit_risk_score",
            "affordability_score",
            "repayment_probability",
        ],
        "pattern_type": "High-Risk: Creditworthiness Assessment (Annex III, area 5(b))",
        "historical_context": (
            "The EU AI Act classifies AI systems evaluating creditworthiness or "
            "credit scoring as HIGH-RISK (Annex III, area 5(b)), except for systems "
            "designed to detect financial fraud. This reflects decades of documented "
            "discrimination in lending, from US redlining (Rothstein, 2017) to "
            "European postcode scoring (SCHUFA/BGH ruling, 2024). The CJEU ruling "
            "in Case C-634/21 (SCHUFA, 2023) clarified that automated credit scoring "
            "falls under GDPR Art. 22. Combined with the AI Act, credit scoring AI "
            "must implement risk management, data governance, transparency, and human "
            "oversight. The European Banking Authority (EBA) has issued guidelines "
            "on AI in credit risk assessment."
        ),
        "affected_groups": [
            "Ethnic minorities",
            "Immigrants",
            "Low-income individuals",
            "Young adults",
            "Women",
            "Residents of disadvantaged areas",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Full conformity assessment required",
            "Test for disparate impact across ethnicity, gender, age, geography",
            "Provide meaningful explanations per GDPR Art. 22 + AI Act Art. 13",
            "Do not use postcode/neighbourhood scoring as sole credit factor",
            "Reference: EU AI Act Annex III area 5(b); CJEU C-634/21; EBA guidelines",
        ],
    },
    # HIGH-RISK: EDUCATION ACCESS & ASSESSMENT (Annex III, area 3)
    "euaia_education": {
        "keywords": [
            "admission_score",
            "admission_decision",
            "enrollment_decision",
            "student_assessment",
            "automated_grading",
            "ai_grading",
            "learning_analytics",
            "student_risk",
            "dropout_prediction",
            "dropout_risk",
            "student_profiling",
            "academic_risk",
            "plagiarism_score",
            "cheating_detection",
            "proctoring_score",
            "exam_monitoring",
            "student_monitoring",
            "learning_path",
        ],
        "pattern_type": "High-Risk: Education & Training (Annex III, area 3)",
        "historical_context": (
            "The EU AI Act classifies AI systems determining access to or admission "
            "to educational institutions, evaluating learning outcomes, assessing "
            "appropriate education levels, and monitoring prohibited student behaviour "
            "during tests as HIGH-RISK (Annex III, area 3). The 2020 UK A-Level scandal "
            "(Ofqual) is the defining cautionary tale: an algorithm used school-level "
            "historical performance to moderate grades, systematically disadvantaging "
            "state school and ethnic minority students. Automated proctoring systems "
            "have shown racial bias (darker skin tones flagged more frequently). "
            "Learning analytics and dropout prediction models risk encoding "
            "socioeconomic disadvantage as individual deficiency."
        ),
        "affected_groups": [
            "Students from deprived areas",
            "Ethnic minority students",
            "Students with disabilities",
            "First-generation students",
            "Non-native language speakers",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Implement conformity assessment procedures",
            "Test proctoring and monitoring systems for racial and disability bias",
            "Do not use school-level performance to penalise individual students",
            "Ensure human review of all automated admission and grading decisions",
            "Reference: EU AI Act Annex III area 3; Ofqual Review (2020)",
        ],
    },
    # HIGH-RISK: ESSENTIAL SERVICES, benefits and insurance (Annex III, area 5(a))
    "euaia_essential_services": {
        "keywords": [
            "benefit_eligibility",
            "benefits_assessment",
            "welfare_eligibility",
            "service_eligibility",
            "entitlement_score",
            "eligibility_score",
            "insurance_risk",
            "insurance_pricing",
            "insurance_score",
            "health_insurance_risk",
            "life_insurance_risk",
            "actuarial_score",
            "risk_premium",
            "claims_prediction",
            "emergency_triage",
            "emergency_priority",
            "dispatch_priority",
            "triage_score",
        ],
        "pattern_type": "High-Risk: Essential Services & Insurance (Annex III, area 5(a))",
        "historical_context": (
            "The EU AI Act classifies AI systems used by public authorities to "
            "evaluate eligibility for public benefits and services (including "
            "allocation, reduction, revocation, or recovery), evaluate "
            "creditworthiness, and assess risk/pricing in health and life insurance "
            "as HIGH-RISK (Annex III, area 5). This reflects documented discrimination "
            "in welfare algorithms (Dutch Toeslagenaffaire, UK Universal Credit) and "
            "insurance pricing (gender-based pricing banned by CJEU Test-Achats ruling "
            "C-236/09, 2011). Emergency triage and dispatch AI is also high-risk, "
            "requiring conformity assessment to prevent discriminatory prioritisation."
        ),
        "affected_groups": [
            "Low-income individuals",
            "Ethnic minorities",
            "Immigrants",
            "Persons with disabilities",
            "Elderly",
            "Single parents",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Full conformity assessment required",
            "Audit benefit eligibility algorithms for ethnic and socioeconomic bias",
            "Do not use protected attributes for insurance risk pricing",
            "Ensure human oversight for benefit allocation and revocation decisions",
            "Reference: EU AI Act Annex III area 5; CJEU Test-Achats (2011)",
        ],
    },
    # HIGH-RISK: LAW ENFORCEMENT (Annex III, area 6)
    "euaia_law_enforcement": {
        "keywords": [
            "risk_assessment_le",
            "reoffending_risk",
            "recidivism_risk",
            "crime_risk_score",
            "threat_assessment",
            "dangerousness_score",
            "polygraph",
            "lie_detection",
            "deception_detection",
            "evidence_reliability",
            "criminal_profiling",
            "suspect_profiling",
            "victimisation_risk",
            "victim_risk",
            "offender_risk",
        ],
        "pattern_type": "High-Risk: Law Enforcement (Annex III, area 6)",
        "historical_context": (
            "The EU AI Act classifies AI systems used in law enforcement for "
            "individual risk assessment, polygraph/lie detection, evidence reliability "
            "evaluation, crime victim risk assessment, and criminal profiling as "
            "HIGH-RISK (Annex III, area 6). Note: purely profile-based crime "
            "prediction without objective facts is PROHIBITED under Art. 5(1)(d). "
            "The COMPAS recidivism algorithm (US) was shown to have double the false "
            "positive rate for Black defendants (ProPublica, 2016). UK Gangs Matrix "
            "and Dutch CAS show similar European biases. The AI Act requires "
            "fundamental rights impact assessments for law enforcement AI."
        ),
        "affected_groups": [
            "Ethnic minorities",
            "Black communities",
            "Roma",
            "Young men",
            "Low-income neighbourhoods",
            "Immigrants",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Conformity assessment + FRIA required",
            "Purely profile-based crime prediction is PROHIBITED (Art. 5(1)(d))",
            "Audit risk scores for racial and ethnic disproportionality",
            "Conduct fundamental rights impact assessment before deployment",
            "Reference: EU AI Act Annex III area 6; Art. 5(1)(d); Art. 27",
        ],
    },
    # HIGH-RISK: MIGRATION & BORDER CONTROL (Annex III, area 7)
    "euaia_migration_border": {
        "keywords": [
            "asylum_assessment",
            "asylum_score",
            "visa_decision",
            "visa_risk",
            "border_risk",
            "immigration_risk",
            "travel_risk",
            "irregular_migration",
            "migration_risk",
            "residence_permit_decision",
            "deportation_risk",
            "return_decision",
            "health_risk_migration",
            "security_risk_migration",
            "document_verification",
            "travel_document",
            "etias",
        ],
        "pattern_type": "High-Risk: Migration & Border Control (Annex III, area 7)",
        "historical_context": (
            "The EU AI Act classifies AI systems used for asylum application "
            "examination, visa and residence permit decisions, polygraphs at borders, "
            "irregular migration risk assessment, and health/security risk screening "
            "as HIGH-RISK (Annex III, area 7). The EU's ETIAS (European Travel "
            "Information and Authorisation System) and the Entry-Exit System raise "
            "significant profiling concerns. Frontex's risk analysis algorithms have "
            "been criticised by the EU Ombudsman and FRA for potential ethnic profiling. "
            "The Act specifically requires that these systems do not discriminate based "
            "on nationality, ethnicity, or religion."
        ),
        "affected_groups": [
            "Asylum seekers",
            "Refugees",
            "Visa applicants",
            "Third-country nationals",
            "Muslim-majority country nationals",
            "African nationals",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Full conformity assessment required",
            "Must not discriminate based on nationality, ethnicity, or religion",
            "Conduct fundamental rights impact assessment (Art. 27)",
            "Ensure human review for all asylum and visa decisions",
            "Reference: EU AI Act Annex III area 7; FRA reports; EU Ombudsman",
        ],
    },
    # HIGH-RISK: JUDICIAL / DEMOCRATIC PROCESSES (Annex III, area 8)
    "euaia_justice_democracy": {
        "keywords": [
            "sentencing_score",
            "judicial_prediction",
            "case_prediction",
            "legal_outcome_prediction",
            "recidivism_prediction",
            "bail_score",
            "pretrial_risk",
            "dispute_resolution_score",
            "adr_score",
            "election_influence",
            "voter_targeting",
            "political_profiling",
            "voter_score",
            "political_ad_targeting",
            "campaign_targeting",
            "disinformation_score",
        ],
        "pattern_type": "High-Risk: Justice & Democratic Processes (Annex III, area 8)",
        "historical_context": (
            "The EU AI Act classifies AI systems used for researching and interpreting "
            "facts and applying law, alternative dispute resolution, and influencing "
            "elections or voting behaviour as HIGH-RISK (Annex III, area 8). This "
            "reflects concerns about algorithmic justice (France banned predictive "
            "analytics of judicial decisions in 2019) and election manipulation "
            "(Cambridge Analytica scandal, 2018). The prohibition on election "
            "influence excludes tools that do not directly interact with individuals "
            "(e.g., campaign logistics). AI-assisted judicial decisions must maintain "
            "judicial independence and ensure the right to a fair trial (ECHR Art. 6)."
        ),
        "affected_groups": [
            "Defendants",
            "Litigants",
            "Voters",
            "Political minorities",
            "Marginalised communities",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "HIGH-RISK under EU AI Act: Conformity assessment required",
            "Judicial AI must preserve judicial independence and fair trial rights",
            "Election-related AI targeting is high-risk; manipulation is prohibited",
            "French precedent: predictive analytics of judicial decisions banned (2019)",
            "Reference: EU AI Act Annex III area 8; ECHR Art. 6; Art. 5(1)(a)",
        ],
    },
    # SWISS-SPECIFIC AI FAIRNESS PATTERNS
    # RESIDENCE PERMIT SYSTEM (CH)
    "swiss_permit_system": {
        "keywords": [
            "aufenthaltsbewilligung",
            "permit_type",
            "permit_status",
            "residence_permit",
            "bewilligung",
            "ausweis",
            "ausweis_b",
            "ausweis_c",
            "ausweis_l",
            "ausweis_f",
            "ausweis_n",
            "ausweis_s",
            "niederlassung",
            "niederlassungsbewilligung",
            "grenzgänger",
            "grenzgaenger",
            "frontalier",
            "aufenthalt",
            "kurzaufenthalt",
            "vorläufig_aufgenommen",
            "vorlaeufig_aufgenommen",
        ],
        "pattern_type": "Swiss Residence Permit Discrimination",
        "historical_context": (
            "Switzerland's tiered permit system (B=temporary, C=permanent, "
            "L=short-term, F=provisionally admitted, N=asylum seeker, S=protection "
            "status) creates a legal hierarchy that strongly correlates with nationality "
            "and ethnicity. Permit type affects access to housing, credit, employment, "
            "and insurance. Studies by the Swiss Forum for Migration (SFM) show that "
            "permit status is used as a proxy for integration and trustworthiness, "
            "systematically disadvantaging non-EU nationals, refugees, and provisionally "
            "admitted persons. Using permit type in algorithms perpetuates a system where "
            "legal status becomes a proxy for national origin."
        ),
        "affected_groups": [
            "Non-EU nationals",
            "Refugees",
            "Provisionally admitted persons",
            "Asylum seekers",
            "Third-country nationals",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Do not use permit type as a feature in credit, housing, or employment algorithms",
            "Audit whether permit status proxies for nationality or ethnicity",
            "Comply with Swiss Federal Act on Foreign Nationals (AIG/LEI) anti-discrimination provisions",
            "Consider that permit status reflects immigration policy, not individual risk",
            "Reference: SFM/Uni Neuchâtel studies; Federal Commission against Racism (EKR/CFR)",
        ],
    },
    # BETREIBUNGSREGISTER / DEBT COLLECTION REGISTER (CH)
    "swiss_betreibung": {
        "keywords": [
            "betreibung",
            "betreibungsregister",
            "betreibungsauskunft",
            "betreibungsauszug",
            "schuldbetreibung",
            "verlustschein",
            "zahlungsbefehl",
            "pfändung",
            "pfaendung",
            "konkurs",
            "betreibungsamt",
            "inkasso",
            "debt_register",
            "debt_collection",
            "collection_record",
            "poursuites",
            "office_poursuites",
            "extrait_poursuites",
            "esecuzione",
        ],
        "pattern_type": "Swiss Debt Register Discrimination",
        "historical_context": (
            "Switzerland's Betreibungsregister (debt collection register) is uniquely "
            "powerful: anyone can file a Betreibung (debt enforcement request) against "
            "anyone, even for disputed or invalid claims. Entries remain visible and "
            "are routinely demanded by landlords, employers, and insurers. The system "
            "disproportionately affects immigrants (who may not understand the 10-day "
            "Rechtsvorschlag/opposition deadline), low-income individuals, and those "
            "in precarious employment. A single entry, even if successfully contested, "
            "can block access to housing and employment for years. The Swiss Mieterverband "
            "and consumer protection organisations have documented its discriminatory impact."
        ),
        "affected_groups": [
            "Immigrants",
            "Low-income individuals",
            "Young adults",
            "Non-German/French/Italian speakers",
            "Gig economy workers",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Distinguish between valid judgments and contested/withdrawn Betreibungen",
            "Do not treat a Betreibungsauszug as equivalent to creditworthiness assessment",
            "Consider that entries may reflect unfamiliarity with the Swiss legal system",
            "Allow applicants to explain entries before making automated decisions",
            "Reference: Mieterverband Schweiz; Konsumentenschutz; SchKG Art. 8a",
        ],
    },
    # HOUSING MARKET DISCRIMINATION (CH)
    "swiss_housing_discrimination": {
        "keywords": [
            "wohnungsbewerbung",
            "mieter",
            "mietgesuch",
            "wohnungsgesuch",
            "nachname",
            "familienname",
            "herkunft",
            "herkunftsland",
            "applicant_name",
            "tenant_name",
            "tenant_nationality",
            "wohnungsmarkt",
            "mieterspiegel",
            "mietzins",
            "genossenschaft",
            "wohnbaugenossenschaft",
            "gemeinnützig",
        ],
        "pattern_type": "Swiss Housing Market Discrimination",
        "historical_context": (
            "Multiple studies (University of Zurich, University of Neuchâtel) have "
            "documented systematic discrimination in the Swiss rental market. Applicants "
            "with names signalling Balkan, Turkish, or African origin receive 20-50%% "
            "fewer callbacks than identical applicants with Swiss-German names. The tight "
            "housing market (vacancy rates below 1%% in major cities) amplifies this effect. "
            "Online platforms and algorithmic tenant selection tools risk automating these "
            "biases. The Federal Commission against Racism (EKR/CFR) has repeatedly "
            "highlighted housing discrimination as Switzerland's most pressing discrimination "
            "issue alongside the labour market."
        ),
        "affected_groups": [
            "Balkan-origin residents",
            "Turkish-origin residents",
            "African-origin residents",
            "Refugees",
            "Large families",
            "Single parents",
            "Social welfare recipients",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Anonymise applicant names in tenant selection algorithms",
            "Audit automated tenant scoring for disparate impact by nationality/origin",
            "Do not use nationality or permit type as tenant selection criteria",
            "Comply with Swiss anti-discrimination norms (Art. 8 BV/Cst)",
            "Reference: Uni Zürich housing discrimination studies; EKR/CFR reports",
        ],
    },
    # EINBÜRGERUNG / NATURALISATION (CH)
    "swiss_naturalisation": {
        "keywords": [
            "einbürgerung",
            "einbuergerung",
            "naturalisation",
            "naturalization",
            "bürgerrecht",
            "buergerrecht",
            "schweizer_pass",
            "swiss_passport",
            "integration_score",
            "integration_bewertung",
            "integrationsvereinbarung",
            "einbürgerungskommission",
            "einbuergerungskommission",
            "integration_assessment",
            "civic_knowledge",
        ],
        "pattern_type": "Naturalisation Process Bias",
        "historical_context": (
            "Switzerland's naturalisation process is uniquely decentralised: municipalities "
            "(Gemeinden) make the primary decision, historically through public votes "
            "(Gemeindeversammlungen) in some cantons. The Federal Supreme Court ruled "
            "(BGE 129 I 217, 2003 Emmen case) that ballot-box naturalisations violated "
            "the prohibition of discrimination, after applicants from the former Yugoslavia "
            "were systematically rejected while Western European applicants were approved. "
            "Even under the reformed system (since 2018), integration assessments involve "
            "subjective criteria (language, 'Swiss values', social contacts) that can encode "
            "cultural bias. Algorithmic scoring of integration risks automating these biases."
        ),
        "affected_groups": [
            "Balkan-origin applicants",
            "Turkish-origin applicants",
            "Muslim applicants",
            "African-origin applicants",
            "Non-European nationals",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Do not automate integration assessments without human oversight",
            "Audit integration scoring for disparate impact by national origin",
            "Ensure criteria are objective, transparent, and legally defensible",
            "Comply with Swiss Citizenship Act (BüG/LN) Art. 11-14 requirements",
            "Reference: BGE 129 I 217 (Emmen case); BüG revision 2018; EKR/CFR",
        ],
    },
    # HEALTH INSURANCE PREMIUM REGIONS (CH)
    "swiss_health_insurance": {
        "keywords": [
            "prämienregion",
            "praemienregion",
            "premium_region",
            "krankenkasse",
            "krankenversicherung",
            "kvg",
            "lamal",
            "grundversicherung",
            "franchise",
            "selbstbehalt",
            "prämienverbilligung",
            "praemienverbilligung",
            "ipe",
            "health_premium",
            "insurance_region",
            "insurance_zone",
        ],
        "pattern_type": "Swiss Health Insurance Geographic Discrimination",
        "historical_context": (
            "Switzerland's mandatory health insurance (KVG/LAMal) uses premium regions "
            "(Prämienregionen) that create systematic geographic price discrimination. "
            "Urban areas with higher healthcare infrastructure costs have higher premiums, "
            "but these regions also concentrate immigrant and low-income populations. "
            "Premium subsidies (Prämienverbilligung/IPE) vary dramatically by canton, "
            "from generous (GE, VD) to restrictive (LU, AG). The premium region system "
            "effectively creates a postcode lottery for healthcare costs that correlates "
            "with socioeconomic status and migration background. Algorithmic risk "
            "adjustment models (Risikoausgleich) also risk encoding demographic biases."
        ),
        "affected_groups": [
            "Urban low-income residents",
            "Immigrants",
            "Young adults",
            "Large families",
            "Working poor",
        ],
        "risk_level": HistoricalRiskLevel.MEDIUM,
        "recommendations": [
            "Do not use premium region as a proxy for health risk or creditworthiness",
            "Audit risk adjustment models for demographic bias",
            "Consider that premium region correlates with migration background",
            "Ensure supplementary insurance underwriting complies with anti-discrimination law",
            "Reference: BAG/OFSP premium statistics; CSS/SASIS data; KVG Art. 41",
        ],
    },
    # LABOUR MARKET DISCRIMINATION (CH)
    "swiss_labour_discrimination": {
        "keywords": [
            "rav",
            "regionales_arbeitsvermittlungszentrum",
            "orp",
            "arbeitslosenversicherung",
            "alv",
            "stellenmeldepflicht",
            "inländervorrang",
            "inlaendervorrang",
            "profiling_score",
            "arbeitsmarktfähigkeit",
            "arbeitsmarktfaehigkeit",
            "vermittlungsfähigkeit",
            "vermittlungsfaehigkeit",
            "rav_score",
            "placement_score",
            "employability_ch",
        ],
        "pattern_type": "Swiss Employment Service Profiling Bias",
        "historical_context": (
            "Swiss Regional Employment Centres (RAV/ORP) use statistical profiling "
            "to assess jobseekers' reintegration prospects and allocate resources. "
            "Research (University of Lausanne, SECO evaluations) has shown that these "
            "models can disadvantage women (career interruptions for childcare), older "
            "workers (50+), and non-Swiss nationals. The Stellenmeldepflicht (job "
            "reporting obligation) introduced in 2018 created a two-tier labour market "
            "favouring Swiss/EU nationals in professions with high unemployment. "
            "Algorithmic profiling in employment services risks channelling disadvantaged "
            "groups into lower-quality interventions."
        ),
        "affected_groups": [
            "Women",
            "Workers over 50",
            "Non-Swiss nationals",
            "Non-EU nationals",
            "Career changers",
            "Part-time workers",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Audit RAV/ORP profiling models for disparate impact by gender, age, nationality",
            "Do not use nationality as an input to employability scoring",
            "Ensure human caseworker oversight for resource allocation decisions",
            "Monitor Stellenmeldepflicht effects for discriminatory outcomes",
            "Reference: SECO evaluations; Uni Lausanne LIVES studies; AMOSA reports",
        ],
    },
    # GEMEINDE / MUNICIPALITY-LEVEL DATA (CH)
    "swiss_gemeinde_data": {
        "keywords": [
            "gemeinde",
            "gemeinde_code",
            "bfs_nummer",
            "bfs_code",
            "municipality",
            "commune",
            "comune",
            "plz",
            "postleitzahl",
            "ortschaft",
            "wohnort",
            "wohngemeinde",
            "steuergemeinde",
            "tax_municipality",
            "steuerfuss",
            "steuersatz",
            "steuerbelastung",
            "gemeindetypologie",
        ],
        "pattern_type": "Swiss Municipality-Level Discrimination",
        "historical_context": (
            "Switzerland's 2,131 municipalities (Gemeinden) vary enormously in tax rates, "
            "social services, school quality, and demographic composition. The BFS "
            "(Federal Statistical Office) municipality code and Gemeindetypologie encode "
            "detailed socioeconomic information. Low-tax municipalities (e.g., Wollerau SZ, "
            "Freienbach SZ) attract wealthy, predominantly Swiss populations, while urban "
            "centres and agglomeration municipalities have higher immigrant shares and lower "
            "incomes. Using municipality-level data as features, particularly Steuerfuss "
            "(tax multiplier) or Gemeindetypologie, can proxy for wealth, nationality, "
            "and social class. This is the Swiss equivalent of postcode discrimination."
        ),
        "affected_groups": [
            "Residents of urban municipalities",
            "Immigrants",
            "Low-income individuals",
            "Social welfare recipients",
        ],
        "risk_level": HistoricalRiskLevel.HIGH,
        "recommendations": [
            "Avoid using Gemeinde-level features that proxy for socioeconomic status",
            "Do not use Steuerfuss/tax rate as an individual risk indicator",
            "Test for disparate impact when using geographic features in Switzerland",
            "Consider that BFS municipality codes encode demographic composition",
            "Reference: BFS Gemeindetypologie; Swiss Federal Tax Administration; EKR/CFR",
        ],
    },
    # SOZIALHILFE / SOCIAL WELFARE (CH)
    "swiss_sozialhilfe": {
        "keywords": [
            "sozialhilfe",
            "sozialdienst",
            "fürsorge",
            "fuersorge",
            "aide_sociale",
            "assistenza_sociale",
            "skos",
            "csias",
            "social_assistance",
            "welfare_status",
            "sozialhilfequote",
            "ergänzungsleistungen",
            "ergaenzungsleistungen",
            "sozialversicherung",
            "sozialleistung",
        ],
        "pattern_type": "Swiss Social Welfare Stigmatisation",
        "historical_context": (
            "Social welfare (Sozialhilfe) receipt in Switzerland carries severe stigma "
            "and legal consequences unique to the Swiss system. Recipients may face "
            "repayment obligations, and for non-Swiss nationals, welfare receipt can "
            "trigger permit downgrading or expulsion under AIG/LEI Art. 63. Municipalities "
            "have wide discretion in implementation (following SKOS/CSIAS guidelines), "
            "creating a postcode lottery. Studies show that foreign nationals are "
            "overrepresented in Sozialhilfe statistics partly due to structural barriers "
            "(permit restrictions, credential non-recognition). Using welfare status "
            "in algorithms thus encodes both poverty and immigration-related disadvantage."
        ),
        "affected_groups": [
            "Foreign nationals",
            "Refugees",
            "Single parents",
            "Disabled individuals",
            "Working poor",
            "Young adults",
        ],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": [
            "Never use welfare receipt status as a negative signal in algorithms",
            "Recognise that welfare receipt for non-Swiss nationals reflects structural barriers",
            "Do not share welfare data with immigration authorities via algorithmic systems",
            "Comply with data protection (nDSG) when processing social welfare data",
            "Reference: SKOS/CSIAS guidelines; BFS Sozialhilfestatistik; AIG Art. 63",
        ],
    },
}


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------
#
# Every keyword above used to be matched as a SUBSTRING of the column name.
# 34 of the 635 keywords are five characters or fewer, and on an ordinary
# manufacturing table that invented SEVEN discrimination findings. Measured
# 2026-09-10, on a frame of aroma_score / award_amount / valve_pressure /
# product_name / singapore_office / pipeline_id / iris_diameter:
#
#   aroma_score      -> risk=critical  "Roma and Traveller Discrimination"
#   iris_diameter    -> risk=critical  "Biometric Identification Bias"
#   award_amount     -> risk=high      "Geographic Redlining"
#   valve_pressure   -> risk=high      "Swiss Employment Service Profiling"
#   product_name     -> risk=high      "Name-based Discrimination"
#   pipeline_id      -> risk=medium    "Swiss Health Insurance Discrimination"
#   singapore_office -> risk=medium    "Employment Gap Penalty"
#
# An audit report asserting CRITICAL Roma discrimination because a column is
# called aroma_score destroys the credibility of every other finding in it.
#
# Matching is now WHOLE-TOKEN and, for multi-word keywords, contiguous. That
# alone removes five of the seven (roma/ward/alv/ipe/gap are not tokens of
# those names) and keeps the compound spellings working, because the
# catalogue already carries them: 'zipcode' matches the keyword 'zipcode',
# 'plz_code' matches the keyword 'plz' as a whole token.
#
# The remaining two, product_name and iris_diameter, DO contain the keyword
# as a whole token; 'name' and 'iris' are ordinary English words. Those
# keywords are listed below as ambiguous: they count only when the column is
# exactly that word, or when every OTHER token in the column name is
# person/demographic context. 'employee_name' and 'iris_scan' still match;
# 'product_name' and 'iris_diameter' do not.
#
# Deliberately NOT listed as ambiguous: the abbreviations (alv, orp, ipe,
# nfa, apl, hlm, rav). Whole-token matching already handles them -- they were
# only ever found inside longer words -- and requiring context for them would
# lose the genuine Swiss/EU columns they exist to catch.
_AMBIGUOUS_KEYWORDS = frozenset(
    {
        "block",
        "district",
        "donor",
        "faith",
        "gap",
        "iris",
        "name",
        "nomad",
        "tract",
        "veil",
        "ward",
    }
)

# Tokens that, standing next to an ambiguous keyword, make it read as the
# protected concept rather than as ordinary vocabulary.
_QUALIFYING_CONTEXT_TOKENS = frozenset(
    {
        # the person the row is about
        "applicant",
        "borrower",
        "candidate",
        "citizen",
        "client",
        "customer",
        "driver",
        "employee",
        "employer",
        "family",
        "father",
        "first",
        "full",
        "given",
        "guardian",
        "holder",
        "household",
        "individual",
        "last",
        "maiden",
        "member",
        "middle",
        "mother",
        "parent",
        "passenger",
        "patient",
        "people",
        "person",
        "personal",
        "resident",
        "spouse",
        "staff",
        "student",
        "subject",
        "sur",
        "tenant",
        "user",
        "worker",
        # concept context
        "biometric",
        "career",
        "census",
        "cv",
        "electoral",
        "employment",
        "ethnic",
        "ethnicity",
        "identification",
        "job",
        "minority",
        "postal",
        "recognition",
        "religion",
        "religious",
        "resume",
        "scan",
        "school",
        "template",
        "work",
    }
)


def _keyword_matches_column(column: str, keyword: str) -> bool:
    """Whole-token keyword match, with a context rule for ambiguous words."""
    col_tokens = name_token_list(column)
    kw_tokens = name_token_list(keyword)
    if not kw_tokens or not tokens_contain(col_tokens, kw_tokens):
        return False

    if len(kw_tokens) != 1 or kw_tokens[0] not in _AMBIGUOUS_KEYWORDS:
        return True

    # Ambiguous single word: it must BE the column, or every other token in
    # the column must be person/demographic context.
    others = [t for t in col_tokens if t != kw_tokens[0]]
    if not others:
        return True
    return all(t in _QUALIFYING_CONTEXT_TOKENS for t in others)


def _match_keyword(column: str, keywords: List[str]) -> Optional[str]:
    """First keyword in ``keywords`` that matches ``column``, or None.

    Longest keyword first, so a qualified spelling ('employment_gap') is
    reported in preference to the bare ambiguous one ('gap').
    """
    for keyword in sorted(keywords, key=lambda k: (-len(name_token_list(k)), -len(k))):
        if _keyword_matches_column(column, keyword):
            return keyword
    return None


def detect_historical_patterns(
    df: pd.DataFrame,
    *,
    protected_attributes: Optional[List[str]] = None,
    custom_patterns: Optional[Dict[str, Dict]] = None,
    min_confidence: float = 0.3,
    include_value_analysis: bool = True,
) -> List[HistoricalPatternResult]:
    """
    Detect features that may encode historical discrimination patterns.

    This function analyzes dataset columns to identify features linked to
    known patterns of historical discrimination (redlining, segregation,
    employment discrimination, etc.).

    Args:
        df: DataFrame to analyze
        protected_attributes: Optional list of known protected attribute columns
                            (used for correlation analysis)
        custom_patterns: Optional dictionary of custom historical patterns to check
                        (format matches HISTORICAL_RISK_PATTERNS)
        min_confidence: Minimum confidence threshold for reporting findings
        include_value_analysis: Whether to analyze column values for additional signals

    Returns:
        List of HistoricalPatternResult objects, sorted by risk level.

        When ``protected_attributes`` is given, each finding's ``evidence`` may
        carry two keys. ``protected_correlations`` holds the correlations that
        were MEASURED, over the rows where both columns were observed.
        ``protected_correlations_not_assessable`` names every attribute whose
        correlation could NOT be taken, with the reason: those attributes are
        absent from ``protected_correlations`` and never raise ``confidence``,
        and absent is not the same as uncorrelated. A single UserWarning at the
        end of the call repeats them, for callers who watch warnings instead of
        payloads.

    Example:
        >>> results = detect_historical_patterns(df)
        >>> for r in results:
        ...     print(f"{r.feature}: {r.risk_level.value} - {r.pattern_type}")
        ...     print(f"  Context: {r.historical_context[:100]}...")

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: detect_historical_patterns. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """
    results = []
    # feature -> {protected attribute: why its correlation could not be taken}.
    # Carried on each finding's `evidence` AND warned about once at the end, so
    # it survives both a reader who only looks at the payload and one who only
    # watches warnings.
    correlations_not_assessable: Dict[str, Dict[str, str]] = {}

    all_patterns = dict(HISTORICAL_RISK_PATTERNS)
    if custom_patterns:
        all_patterns.update(custom_patterns)

    for col in df.columns:
        for pattern_name, pattern_info in all_patterns.items():
            matched_keyword = _match_keyword(col, pattern_info.get("keywords", []))

            if matched_keyword:
                confidence = _calculate_historical_confidence(
                    df, col, pattern_info, matched_keyword, include_value_analysis
                )

                if confidence >= min_confidence:
                    evidence = {
                        "matched_keyword": matched_keyword,
                        "column_name": col,
                        "n_unique_values": df[col].nunique(),
                        "sample_values": df[col].dropna().unique()[:5].tolist(),
                    }

                    if protected_attributes:
                        correlations = _compute_protected_correlations(
                            df, col, protected_attributes
                        )
                        if correlations:
                            evidence["protected_correlations"] = dict(correlations)
                            max_corr = max(abs(c) for c in correlations.values())
                            if max_corr > 0.3:
                                confidence = min(1.0, confidence + 0.2)
                        # THREE STATES. An attribute whose correlation could not
                        # be taken is not an attribute that correlates weakly:
                        # it never reaches the confidence bump above, and if it
                        # is merely ABSENT from `protected_correlations` the
                        # reader cannot tell which of the two happened. Named
                        # here, on the evidence a reader already looks at.
                        unassessable = getattr(correlations, "not_assessable", {})
                        if unassessable:
                            evidence["protected_correlations_not_assessable"] = dict(unassessable)
                            correlations_not_assessable[col] = dict(unassessable)

                    result = HistoricalPatternResult(
                        feature=col,
                        risk_level=pattern_info.get("risk_level", HistoricalRiskLevel.MEDIUM),
                        pattern_type=pattern_info.get("pattern_type", "Unknown"),
                        description=f"Column '{col}' matches historical pattern '{pattern_name}' "
                        f"(matched keyword: '{matched_keyword}')",
                        historical_context=pattern_info.get("historical_context", ""),
                        affected_groups=pattern_info.get("affected_groups", []),
                        confidence=confidence,
                        recommendations=pattern_info.get("recommendations", []),
                        evidence=evidence,
                    )
                    results.append(result)
                    break  # Only match one pattern per column

    risk_order = {
        HistoricalRiskLevel.CRITICAL: 0,
        HistoricalRiskLevel.HIGH: 1,
        HistoricalRiskLevel.MEDIUM: 2,
        HistoricalRiskLevel.LOW: 3,
        HistoricalRiskLevel.NONE: 4,
    }
    results.sort(key=lambda x: (risk_order[x.risk_level], -x.confidence))

    if correlations_not_assessable:
        warnings.warn(
            "detect_historical_patterns: no correlation with a protected attribute "
            "could be taken for "
            + "; ".join(
                f"{feature!r} ({', '.join(f'{a}: {why}' for a, why in reasons.items())})"
                for feature, reasons in list(correlations_not_assessable.items())[:5]
            )
            + (" ..." if len(correlations_not_assessable) > 5 else "")
            + ". Those attributes are absent from 'protected_correlations' and did "
            "not raise the reported confidence; absent is not the same as "
            "uncorrelated. See evidence['protected_correlations_not_assessable'].",
            UserWarning,
            stacklevel=2,
        )

    return results


def _calculate_historical_confidence(
    df: pd.DataFrame,
    column: str,
    pattern_info: Dict,
    matched_keyword: str,
    include_value_analysis: bool,
) -> float:
    """Calculate confidence score for historical pattern detection."""
    confidence = 0.0

    col_lower = column.lower()
    if col_lower == matched_keyword or col_lower == matched_keyword.replace("_", " "):
        confidence += 0.6  # Exact match
    elif len(matched_keyword) >= 6:
        confidence += 0.5  # Long keyword partial match
    else:
        confidence += 0.4  # Short keyword match

    risk_level = pattern_info.get("risk_level", HistoricalRiskLevel.MEDIUM)
    if risk_level == HistoricalRiskLevel.CRITICAL:
        confidence += 0.15
    elif risk_level == HistoricalRiskLevel.HIGH:
        confidence += 0.1

    if include_value_analysis:
        try:
            series = df[column]
            n_unique = series.nunique()

            # Geographic patterns often have moderate cardinality
            if "geographic" in pattern_info.get("pattern_type", "").lower():
                if 10 <= n_unique <= 1000:
                    confidence += 0.1

            # Name patterns often have high cardinality
            if "name" in pattern_info.get("pattern_type", "").lower():
                if n_unique > 100:
                    confidence += 0.1

        except Exception:
            logging.getLogger(__name__).debug(
                "optional computation failed; skipping", exc_info=True
            )

    return min(1.0, confidence)


# Minimum number of rows where BOTH columns are observed before a correlation
# between them is reported at all. Below it the answer is "could not check",
# recorded by name in `_CorrelationResult.not_assessable`, never a 0.0.
_MIN_CORRELATION_OVERLAP = 10


class _CorrelationResult(Dict[str, float]):
    """Correlations that can also say which attributes could NOT be correlated.

    House style, matching ``_ScanResult`` / ``_IntersectionalResult`` in
    ``evaluation.vfairness_metrics.discovery``: this stays a plain dict for every
    existing caller, and carries the could-not-check reasons for the ones that
    ask. It exists because an ABSENT correlation and a WEAK correlation are
    different answers, and the caller raises its reported confidence on the
    strength of the first kind while reading the second kind as reassurance.
    """

    not_assessable: Dict[str, str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.not_assessable = {}


def _encode_for_correlation(series: pd.Series) -> np.ndarray:
    """Encode one column as floats, LEAVING MISSING VALUES MISSING.

    BGL-U01 (2026-09-17). Non-numeric columns were encoded with
    ``pd.Categorical(series).codes``, which hands every missing value the integer
    sentinel ``-1``. ``pd.isna`` on an integer code array is ``False`` everywhere,
    so the overlap mask below could not drop those rows and the "correlation" was
    taken over rows where the column had never been observed: it measured the
    MISSINGNESS PATTERN and reported it as a property of the column.

    Measured at the public entry (``detect_historical_patterns``) on 400 rows
    whose ``zipcode`` was recorded only for white applicants, so that among the
    rows where zipcode WAS observed race is constant and no zipcode/race
    correlation exists to be taken at all (this same function answers ``{}`` when
    handed exactly those rows): ``evidence['protected_correlations']`` came back
    ``{'race': 0.8115}``, that number cleared the 0.3 gate in the caller, and the
    reported ``confidence`` rose from 0.80 to 1.0 with nothing on any channel
    saying the column was half empty.
    """
    if not isinstance(series, pd.Series):
        # `df[name]` hands back a DataFrame when the frame carries that column
        # name twice, and `pd.Categorical` accepts it and returns codes of the
        # WRONG LENGTH rather than raising. Refuse it by name instead.
        raise TypeError(
            f"expected a single column, got {type(series).__name__}; the frame "
            "carries this column name more than once"
        )
    if pd.api.types.is_numeric_dtype(series):  # robust to pandas-3 str dtype
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    codes = pd.Categorical(series).codes.astype(float)
    # -1 is pandas' "not in any category", i.e. missing. It is not category -1.
    codes[codes < 0] = np.nan
    return codes


def _compute_protected_correlations(
    df: pd.DataFrame,
    column: str,
    protected_attributes: List[str],
) -> Dict[str, float]:
    """Correlate a column with protected attributes over OBSERVED rows only.

    Returns attribute -> Pearson r, computed over the rows where both values
    were actually observed. An attribute whose correlation could not be taken is
    ABSENT from that mapping and named in ``.not_assessable`` with the reason;
    it is never given a 0.0 stand-in, because 0.0 on this scale means "measured,
    and unrelated" and is the most reassuring answer available.
    """
    correlations = _CorrelationResult()

    try:
        series_encoded = _encode_for_correlation(df[column])
    except Exception as exc:
        logging.getLogger(__name__).debug("optional computation failed; skipping", exc_info=True)
        for attr in protected_attributes:
            if attr != column:
                correlations.not_assessable[attr] = (
                    f"column {column!r} could not be encoded ({type(exc).__name__}: {exc})"
                )
        return correlations

    for attr in protected_attributes:
        if attr == column:
            continue
        if attr not in df.columns:
            correlations.not_assessable[attr] = "not a column of this frame"
            continue

        try:
            attr_encoded = _encode_for_correlation(df[attr])
        except Exception as exc:
            correlations.not_assessable[attr] = (
                f"column {attr!r} could not be encoded ({type(exc).__name__}: {exc})"
            )
            continue

        mask = ~(np.isnan(series_encoded) | np.isnan(attr_encoded))
        n_overlap = int(mask.sum())
        if n_overlap < _MIN_CORRELATION_OVERLAP:
            correlations.not_assessable[attr] = (
                f"only {n_overlap} of {len(mask)} row(s) have both {column!r} and "
                f"{attr!r} observed, below the minimum of "
                f"{_MIN_CORRELATION_OVERLAP} needed to correlate them"
            )
            continue

        left = series_encoded[mask]
        right = attr_encoded[mask]
        # np.ptp, never `np.var(...) == 0.0`: an accumulated statistic is exactly
        # zero only for the round numbers that happen to be powers of two.
        if np.ptp(left) == 0 or np.ptp(right) == 0:
            constant = column if np.ptp(left) == 0 else attr
            correlations.not_assessable[attr] = (
                f"{constant!r} is constant over the {n_overlap} row(s) where both "
                f"{column!r} and {attr!r} are observed, so their correlation is "
                "undefined"
            )
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(left, right)[0, 1]

        if np.isnan(corr):
            correlations.not_assessable[attr] = (
                f"the correlation of {column!r} with {attr!r} came back undefined "
                f"over the {n_overlap} row(s) where both are observed"
            )
            continue

        correlations[attr] = round(float(corr), 4)

    return correlations


# The strings ABSENCE MINTS. ``str()`` of an absent value is content: str(None) is
# 'None', str(float('nan')) is 'nan', str(pd.NA) is '<NA>' and str(pd.NaT) is
# 'NaT'. A CSV export writes those, and a JSON one writes 'null'. Each of them
# then reads as a real label, and in a geographic column a real label is a real
# PLACE. Deliberately tight: only the spellings an absent value produces, plus the
# blank and whitespace-only string. 'NA' alone is NOT here, because in a
# geographic column it can be North America.
_MINTED_ABSENCE_STRINGS = frozenset({"", "none", "nan", "<na>", "nat", "null"})


def _absent_label_mask(series: "pd.Series") -> "pd.Series":
    """True for every row whose label is absent through ANY of its doors.

    Six doors, and ``str(x)`` on an absent value mints a seventh through each:
    None, float nan, pd.NA, pd.NaT, the blank string, whitespace only, and the
    literal spellings in :data:`_MINTED_ABSENCE_STRINGS`.
    """
    try:
        text = series.astype("string").fillna("").str.strip().str.lower()
        return series.isna() | text.isin(_MINTED_ABSENCE_STRINGS)
    except Exception:  # noqa: BLE001
        # Never a false "everything is absent" from a dtype this cannot render;
        # isna alone is always available and is the conservative half.
        return series.isna()


def check_geographic_redlining_risk(
    df: pd.DataFrame,
    geographic_column: str,
    outcome_column: Optional[str] = None,
    demographic_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Specialized check for geographic redlining patterns.

    Analyzes a geographic column for patterns consistent with historical
    redlining by examining outcome disparities across geographic units
    and their correlation with demographic composition.

    Args:
        df: DataFrame to analyze
        geographic_column: Column containing geographic identifiers
        outcome_column: Optional column with outcomes to analyze
        demographic_column: Optional column with demographic information

    Returns:
        Dictionary with redlining risk analysis results.

        ``redlining_risk_score`` is three-state. It is a float when at least one
        of the three components below was measured, and ``None`` when not one of
        them was, which is UNMEASURED and not a score of zero. Whichever it is,
        ``components_not_assessed`` names every requested component that could
        not be measured and why, ``components_partially_assessed`` names every
        component that WAS measured but only over part of the data, and
        ``assessment_coverage`` is ``"complete"``, ``"partial"`` or
        ``"not_assessed"``. The score is a SUM, so a component that could not be
        taken withholds its contribution exactly like a component that was taken
        and came back clean, and only these keys tell those apart.

        ``outcome_analysis['outcome_range']`` is itself three-state: a float when
        at least two geographic units carry a finite mean, and ``None`` when
        fewer do, because a range is a comparison and one unit is not one.
        ``n_geographic_units_with_a_finite_mean`` states how many units the range
        was taken over.

        The ROW axis is disclosed beside the unit axis, because the exclusion
        happens at row level and a count of units cannot express it. A row whose
        geography is absent through ANY of its doors (None, float nan, ``pd.NA``,
        ``pd.NaT``, the blank or whitespace-only string, or a literal spelling of
        absence such as ``'None'``, which ``str()`` mints and which used to become
        a geographic unit of its own) belongs to no geographic unit. Those rows are
        counted in ``n_rows_with_no_readable_geography``, the rows that were graded
        in ``n_rows_with_a_readable_geography``, and their exclusion is recorded in
        ``components_partially_assessed`` so ``assessment_coverage`` cannot read
        ``"complete"`` over part of a file.
    """
    results: Dict[str, Any] = {
        "geographic_column": geographic_column,
        "redlining_risk_score": 0.0,
        "findings": [],
        "recommendations": [],
    }

    if geographic_column not in df.columns:
        results["error"] = f"Column '{geographic_column}' not found"
        results["redlining_risk_score"] = None
        results["assessment_coverage"] = NOT_ASSESSED
        results["components_not_assessed"] = [
            {
                "component": "all",
                "reason": f"column '{geographic_column}' is not present in the DataFrame",
            }
        ]
        return results

    # BGL3 2026-09-27. Each of the three components below contributes to a SUM
    # that starts at 0.0, so a component that could NOT be taken leaves the
    # score reading exactly like a component that was taken and found nothing.
    # Measured on 600 rows keyed by 60 ZIPs whose outcome column was entirely
    # NULL: redlining_risk_score 0.0, findings [], recommendations [], not one
    # warning, while outcome_range was nan and nothing had been compared. Same
    # 0.0 with the outcome column holding 'yes'/'no' strings, where the groupby
    # raised and the old bare `except` logged it at DEBUG and dropped
    # 'outcome_analysis' from the result altogether, and same 0.0 with the
    # geographic column entirely NULL, where there was no geography to read.
    assessed: List[str] = []
    not_assessed: List[Dict[str, str]] = []
    # A component that WAS measured, on part of the data only. Neither of the two
    # lists above can hold it: it is not a refusal and it is not whole coverage.
    partially_assessed: List[Dict[str, str]] = []

    # THE GEOGRAPHIC KEY'S OWN MISSINGNESS, BGL-F5 2026-09-30, and it is TWO
    # doors in one line, one in each direction.
    #
    # (a) SILENTLY DROPPED. `df.groupby(col)` drops rows whose key is NaN and
    #     `Series.nunique()` excludes them too, so those rows left no trace
    #     anywhere in the result. Measured on 60 rows over 6 ZIPs at a flat 0.50
    #     approval rate: score 0.0, assessment_coverage 'complete',
    #     outcome_analysis {'mean_by_geo_std': 0.0, 'outcome_range': 0.0,
    #     'n_geographic_units': 6, 'n_geographic_units_with_a_finite_mean': 6},
    #     not_assessed [], partial [], warnings []. Add 60 MORE rows whose zip is
    #     None and which are 100 percent DENIED, i.e. half the file and the whole
    #     disparity, and the published result is BYTE-IDENTICAL, with no key
    #     anywhere counting rows. The disclosure the earlier fix added counts
    #     UNITS while the exclusion happens at ROW level, so it read complete by
    #     construction.
    #
    # (b) A UNIT THAT DOES NOT EXIST, the mirror door. `str()` of an absent value
    #     is content, so the literal 'None', the blank string and whitespace are
    #     NOT dropped: they become a geographic unit. Measured on the same 6 ZIPs
    #     plus 60 rows whose zip is the string 'None' and which are 100 percent
    #     denied: score 0.4, n_geographic_units 7, 7 of 7 with a finite mean,
    #     outcome_range 0.5, assessment_coverage 'complete', findings ['Large
    #     outcome disparity across geographic units (range: 50.00%). Potential
    #     redlining pattern.'] and ZERO warnings. A redlining finding against a
    #     place that is not a place. Identical for '' and for '   '.
    #
    # Both are closed the same way, because both are the same mistake about what a
    # geography IS: absence is recognised through every one of its doors, those
    # rows are excluded from the units AND counted at ROW level where a reader
    # looks, and their exclusion is declared a partial assessment. The standard is
    # the one detect_temporal_drift sets for its own time column: "their exclusion
    # is a could-not-check, not evidence that they match the rows that were
    # compared".
    geo_absent = _absent_label_mask(df[geographic_column])
    n_rows_no_geo = int(geo_absent.sum())
    n_rows_with_geo = int(len(df)) - n_rows_no_geo
    results["n_rows_with_a_readable_geography"] = n_rows_with_geo
    results["n_rows_with_no_readable_geography"] = n_rows_no_geo
    geo_df = df.loc[~geo_absent.to_numpy()] if n_rows_no_geo else df
    if n_rows_no_geo:
        partially_assessed.append(
            {
                "component": "all",
                "reason": (
                    f"{n_rows_no_geo} of {len(df)} row(s) carry no readable value in "
                    f"'{geographic_column}' (absent, blank, or a literal spelling of "
                    f"absence such as 'None'), so they are in NO geographic unit and every "
                    f"number below is taken over the other {n_rows_with_geo} row(s). Their "
                    f"exclusion is a could-not-check, not evidence that they match the rows "
                    f"that were compared"
                ),
            }
        )
        warnings.warn(
            f"check_geographic_redlining_risk: {partially_assessed[-1]['reason']}. See "
            f"result['n_rows_with_no_readable_geography'].",
            UserWarning,
            stacklevel=2,
        )

    geo_series = geo_df[geographic_column]
    n_unique_geos = geo_series.nunique()

    # Check cardinality (too few or too many geographic units is suspicious)
    if n_unique_geos < 1:
        not_assessed.append(
            {
                "component": "granularity",
                "reason": (
                    f"'{geographic_column}' carries no observation at all "
                    "(0 distinct values), so there is no geography to grade"
                ),
            }
        )
    else:
        assessed.append("granularity")
        if n_unique_geos < 5:
            results["findings"].append(
                f"Very few geographic units ({n_unique_geos}). May be too coarse for fair analysis."
            )
        elif n_unique_geos > 1000:
            results["findings"].append(
                f"Very granular geographic data ({n_unique_geos} units). "
                "High granularity increases redlining risk."
            )
            results["redlining_risk_score"] += 0.3

    if outcome_column and outcome_column in df.columns:
        try:
            # geo_df, not df: a row with no readable geography is in no unit.
            outcome_by_geo = geo_df.groupby(geographic_column)[outcome_column].mean()
            outcome_std = outcome_by_geo.std()
            outcome_range = outcome_by_geo.max() - outcome_by_geo.min()

            # BGL5 2026-09-27. A RANGE IS A COMPARISON, SO IT NEEDS TWO UNITS.
            # `Series.max()` and `Series.min()` SKIP NaN, so the finiteness test
            # below was satisfied by any single observed geographic unit, and the
            # refusal reached only a column that was null in EVERY unit.
            # Measured on 600 rows keyed by 60 ZIPs where exactly one ZIP kept an
            # observed outcome (13 of 600 rows): assessment_coverage 'complete',
            # components_not_assessed [], redlining_risk_score 0.0,
            # outcome_analysis {'mean_by_geo_std': nan, 'outcome_range': 0.0,
            # 'n_geographic_units': 60}, findings [] and not one warning, i.e. a
            # clean all-clear taken over 1 of 60 units. Same output for a frame
            # with a single geographic unit and for two units where one was
            # entirely NULL. It now reports assessment_coverage 'partial' with
            # outcome_disparity in components_not_assessed, outcome_range None,
            # and a warning; the 60-unit control below still scores 0.4 silently.
            n_units_total = int(len(outcome_by_geo))
            n_units_measured = int(outcome_by_geo.notna().sum())
            range_is_a_comparison = n_units_measured >= 2 and bool(np.isfinite(outcome_range))
            if not range_is_a_comparison:
                not_assessed.append(
                    {
                        "component": "outcome_disparity",
                        "reason": (
                            f"the per-geography mean of '{outcome_column}' is finite "
                            f"over {n_units_measured} of {n_units_total} geographic "
                            "unit(s), and a range across geographic units needs at "
                            "least 2, so no between-unit comparison could be taken"
                        ),
                    }
                )
            else:
                assessed.append("outcome_disparity")
                if outcome_range > 0.3:  # Significant disparity
                    results["findings"].append(
                        f"Large outcome disparity across geographic units "
                        f"(range: {outcome_range:.2%}). Potential redlining pattern."
                    )
                    results["redlining_risk_score"] += 0.4
                if n_units_measured < n_units_total:
                    # Measured over a SUBSET of the geography: the widest gap in
                    # the unobserved units cannot be smaller than this one, so the
                    # range is a lower bound and the coverage is partial. Named as
                    # its own state, because calling it "not assessed" would
                    # discard a real comparison.
                    partially_assessed.append(
                        {
                            "component": "outcome_disparity",
                            "reason": (
                                f"the per-geography mean of '{outcome_column}' is "
                                f"finite over {n_units_measured} of {n_units_total} "
                                "geographic unit(s), so the range is a LOWER BOUND "
                                "taken over the units that were observed"
                            ),
                        }
                    )

            results["outcome_analysis"] = {
                "mean_by_geo_std": round(outcome_std, 4),
                # None, not a number, when no between-unit range exists: a
                # rounded 0.0 taken over one unit is the strongest all-clear this
                # field can carry.
                "outcome_range": (round(outcome_range, 4) if range_is_a_comparison else None),
                "n_geographic_units": n_unique_geos,
                "n_geographic_units_with_a_finite_mean": n_units_measured,
            }
        except Exception as exc:
            # Named on the result, not swallowed into a debug log. The one thing
            # this catches in practice is a non-numeric outcome column, which is
            # a real refusal the caller can act on; at DEBUG level it reached
            # nobody and the score read 0.0.
            logging.getLogger(__name__).warning(
                "geographic outcome analysis failed; reporting could-not-check "
                "rather than a score of zero for it",
                exc_info=True,
            )
            not_assessed.append(
                {
                    "component": "outcome_disparity",
                    "reason": (
                        f"the per-geography mean of '{outcome_column}' could not be "
                        f"computed ({type(exc).__name__}: {exc})"
                    ),
                }
            )
    elif outcome_column:
        not_assessed.append(
            {
                "component": "outcome_disparity",
                "reason": f"column '{outcome_column}' is not present in the DataFrame",
            }
        )

    if demographic_column and demographic_column in df.columns:
        try:
            # geo_df, not df, for the same reason as the groupby above.
            corr = _compute_protected_correlations(geo_df, geographic_column, [demographic_column])
            if corr and abs(list(corr.values())[0]) > 0.3:
                results["findings"].append(
                    f"Geographic column correlates with demographic attribute "
                    f"(r={list(corr.values())[0]:.2f}). Indicates segregation pattern."
                )
                results["redlining_risk_score"] += 0.3
                results["demographic_correlation"] = dict(corr)
            # BGL-U01. This risk score is a SUM, so a correlation that could not
            # be taken subtracts 0.3 from it exactly like a correlation that was
            # taken and came back weak. The reader has to be able to tell those
            # apart, and an absent `demographic_correlation` key cannot.
            unassessable = getattr(corr, "not_assessable", {})
            if unassessable:
                results["demographic_correlation_not_assessable"] = dict(unassessable)
            if corr:
                assessed.append("demographic_correlation")
            else:
                not_assessed.append(
                    {
                        "component": "demographic_correlation",
                        "reason": (
                            f"no correlation between '{geographic_column}' and "
                            f"'{demographic_column}' could be taken"
                            + (
                                f": {'; '.join(str(v) for v in unassessable.values())}"
                                if unassessable
                                else ""
                            )
                        ),
                    }
                )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "geographic demographic correlation failed; reporting "
                "could-not-check rather than a score of zero for it",
                exc_info=True,
            )
            not_assessed.append(
                {
                    "component": "demographic_correlation",
                    "reason": (
                        f"the correlation with '{demographic_column}' could not be "
                        f"computed ({type(exc).__name__}: {exc})"
                    ),
                }
            )
    elif demographic_column:
        not_assessed.append(
            {
                "component": "demographic_correlation",
                "reason": f"column '{demographic_column}' is not present in the DataFrame",
            }
        )

    results["components_not_assessed"] = not_assessed
    results["components_partially_assessed"] = partially_assessed
    if not assessed:
        results["assessment_coverage"] = NOT_ASSESSED
    elif not_assessed or partially_assessed:
        results["assessment_coverage"] = "partial"
    else:
        results["assessment_coverage"] = "complete"

    if not assessed:
        # None, not 0.0. Nothing was compared, so there is no score; a 0.0 here
        # is the strongest all-clear this number can give and it would be the
        # only thing most readers look at.
        results["redlining_risk_score"] = None
        warnings.warn(
            f"check_geographic_redlining_risk: not one component of the redlining "
            f"score could be measured for '{geographic_column}' "
            f"({'; '.join(e['reason'] for e in not_assessed)}). Reporting "
            f"redlining_risk_score=None (could not check), NOT 0.0, which would read "
            f"as no redlining risk.",
            UserWarning,
            stacklevel=2,
        )
        return results

    if not_assessed:
        warnings.warn(
            f"check_geographic_redlining_risk: {len(assessed)} of "
            f"{len(assessed) + len(not_assessed)} component(s) of the redlining score "
            f"were measured for '{geographic_column}'; these were NOT: "
            f"{'; '.join(e['component'] + ' (' + e['reason'] + ')' for e in not_assessed)}. "
            f"The score is a SUM, so it is missing their contribution and is a LOWER "
            f"BOUND, not a measurement of the whole. See "
            f"result['components_not_assessed'].",
            UserWarning,
            stacklevel=2,
        )

    if partially_assessed:
        warnings.warn(
            f"check_geographic_redlining_risk: "
            f"{'; '.join(e['component'] + ' (' + e['reason'] + ')' for e in partially_assessed)}. "
            f"That component was measured over PART of the geography only, so the "
            f"redlining_risk_score is a LOWER BOUND for it. See "
            f"result['components_partially_assessed'].",
            UserWarning,
            stacklevel=2,
        )

    results["redlining_risk_score"] = min(1.0, results["redlining_risk_score"])

    if results["redlining_risk_score"] >= 0.5:
        results["recommendations"] = [
            "Consider removing or aggregating geographic feature",
            "Apply fairness constraints if geographic feature is necessary",
            "Test model predictions for geographic disparities",
            "Cross-reference with historical redlining maps if available",
        ]
    elif results["redlining_risk_score"] >= 0.3:
        results["recommendations"] = [
            "Monitor geographic feature for disparate impact",
            "Consider using broader geographic aggregations",
        ]

    return results


# DOMAIN-LEVEL HISTORICAL CONTEXT (precedent banner)
#
# detect_historical_patterns() scans COLUMNS. This adds the orthogonal,
# spec-required DOMAIN banner: "this is a hiring dataset; hiring algorithms
# have documented disparate-impact precedent (citations); observed
# disparities here are consistent with that, not anomalous."
#
# Every citation below is a real, canonical, peer-reviewed or
# regulator-of-record source. Per the platform rule, NOTHING here is
# invented: no fabricated metrics, only the existence of the precedent.

_DOMAIN_HISTORICAL_PRECEDENT: Dict[str, Dict[str, Any]] = {
    "hiring": {
        "label": "Hiring / employment",
        "summary": (
            "Hiring algorithms have documented, repeatable "
            "disparate-impact patterns against women, Black "
            "candidates and older applicants. Observed disparities "
            "in a hiring dataset are typically consistent with these "
            "documented patterns, not anomalous to them."
        ),
        "citations": [
            "Bertrand & Mullainathan (2004), 'Are Emily and Greg More "
            "Employable than Lakisha and Jamal?', American Economic Review",
            "Dastin (2018), Reuters: Amazon scrapped a recruiting tool that "
            "penalised resumes containing the word 'women's'",
            "Raghavan, Barocas, Kleinberg & Levy (2020), 'Mitigating Bias in "
            "Algorithmic Hiring', FAccT",
        ],
        "regulatory": [
            "NYC Local Law 144 (automated employment decision tools, bias audit)",
            "US EEOC Title VII / Uniform Guidelines (four-fifths rule)",
            "EU AI Act Annex III(4): employment is high-risk",
        ],
    },
    "lending": {
        "label": "Credit / lending",
        "summary": (
            "Credit models inherit decades of redlining and "
            "disparate-impact precedent; proxies (ZIP, neighbourhood) "
            "routinely reconstruct race even when race is excluded."
        ),
        "citations": [
            "Hurtado & Sakong (2024) and the long ECOA/Reg-B line; "
            "FHA/HOLC redlining (1930s-): Rothstein (2017), 'The Color of "
            "Law'",
            "Bartlett, Morse, Stanton & Wallace (2022), 'Consumer-Lending "
            "Discrimination in the FinTech Era', Journal of Financial "
            "Economics",
        ],
        "regulatory": [
            "US ECOA / Regulation B (adverse-action reasons)",
            "Fair Housing Act",
            "EU AI Act Annex III(5)(b): creditworthiness is high-risk",
        ],
    },
    "healthcare": {
        "label": "Healthcare",
        "summary": (
            "Health risk algorithms have a landmark documented case "
            "of cost-based proxies systematically under-serving Black "
            "patients at equal sickness."
        ),
        "citations": [
            "Obermeyer, Powers, Vogeli & Mullainathan (2019), 'Dissecting "
            "racial bias in an algorithm used to manage the health of "
            "populations', Science",
        ],
        "regulatory": [
            "EU AI Act Annex III: safety components / essential services",
            "US Section 1557 ACA non-discrimination",
        ],
    },
    "justice": {
        "label": "Criminal justice / recidivism",
        "summary": (
            "Recidivism risk scores have a landmark documented "
            "finding of higher false-positive rates for Black "
            "defendants."
        ),
        "citations": [
            "Angwin, Larson, Mattu & Kirchner (2016), ProPublica, 'Machine Bias' (COMPAS)",
            "Chouldechova (2017), 'Fair prediction with disparate impact', "
            "Big Data: the impossibility result",
        ],
        "regulatory": [
            "EU AI Act Annex III(6): law enforcement is high-risk",
            "US due-process / equal-protection",
        ],
    },
    "education": {
        "label": "Education / admissions",
        "summary": (
            "Admissions and education models carry legacy-preference "
            "and standardised-test disparities that advantage "
            "historically privileged groups."
        ),
        "citations": [
            "Buolamwini & Gebru (2018), 'Gender Shades', PMLR (for any face/photo-derived feature)",
        ],
        "regulatory": ["EU AI Act Annex III(3): education is high-risk"],
    },
    "insurance": {
        "label": "Insurance",
        "summary": (
            "Insurance pricing/underwriting models inherit "
            "geographic and socioeconomic proxies with documented "
            "disparate impact."
        ),
        "citations": [
            "NAIC and state DOI actions on proxy discrimination in "
            "actuarial models; Rothstein (2017), 'The Color of Law'",
        ],
        "regulatory": [
            "State insurance anti-discrimination law",
            "EU AI Act Annex III(5)(c): life/health insurance risk assessment",
        ],
    },
}

_DOMAIN_ALIASES = {
    "employment": "hiring",
    "recruitment": "hiring",
    "recruiting": "hiring",
    "jobs": "hiring",
    "hr": "hiring",
    "credit": "lending",
    "loan": "lending",
    "loans": "lending",
    "finance": "lending",
    "banking": "lending",
    "fintech": "lending",
    "health": "healthcare",
    "medical": "healthcare",
    "clinical": "healthcare",
    "recidivism": "justice",
    "criminal": "justice",
    "policing": "justice",
    "law enforcement": "justice",
    "bail": "justice",
    "admissions": "education",
    "school": "education",
    "university": "education",
    # 'mortgage' is the lending precedent in this very table: the Fair Housing
    # Act / HOLC redlining line under "lending" IS mortgage lending, and the
    # alias was missing beside 'credit', 'loan', 'finance' and 'banking'.
    "mortgage": "lending",
    "mortgages": "lending",
}


def _resolve_domain_key(domain: Optional[str], candidates: Optional[Dict[str, str]] = None) -> str:
    """Resolve a free-text domain to a canonical domain key, whole-token.

    WHOLE-TOKEN, NEVER SUBSTRING, AND LONGEST MATCH WINS. BGL5 2026-09-27. Both
    callers used to fall back to ``if alias in key``, a raw substring test over a
    free-text phrase, and what they return on a match is a citation-backed
    precedent banner. Measured before, with the alias table holding 'hr' and
    'bail': 'chronic disease management' resolved to ``hiring`` and came back
    carrying the Amazon recruiting-tool citation, as did 'threat detection',
    'anthropology research', 'shrinkage analytics' and even 'xhrx'; 'bailout
    underwriting' and 'mortgage bail-in' resolved to ``justice`` and came back
    carrying the COMPAS recidivism finding. After: the first five and 'bailout
    underwriting' resolve to nothing (the caller returns None, which is what its
    docstring promises), 'mortgage bail-in' resolves to ``lending`` because
    'mortgage' is the longer whole-token match, 'health insurance' resolves to
    ``insurance`` rather than ``healthcare`` for the same reason, and 'hr
    analytics', 'online hiring platform' and a bare 'bail' still resolve.

    ``candidates`` defaults to the canonical precedent keys plus the alias table;
    a caller with a narrower set of usable keys passes its own.

    THE TIEBREAK IS ARBITRARY, AND THE CALLER MUST SAY SO. BGL-F5 2026-09-30.
    When two DOCUMENTED domains both match as whole tokens this returns one of
    them decided by character length, and both callers turned that pick into a
    citation-backed precedent with no warning and no third state. Measured before:
    ``domain_historical_context('university hospital clinical trials')`` resolved
    to ``education`` and led its citations with Buolamwini & Gebru's 'Gender
    Shades', a facial-recognition paper, as the documented precedent for a
    clinical-trial audit, because 'university' (10 chars) outranks 'clinical' (8);
    word order did not change it, so it is the tiebreak and not an ordering
    artefact. Also 'criminal justice education program' -> education not justice,
    'medical school admissions' -> education, and 'employment credit screening' ->
    hiring, dropping lending. :func:`_domain_terms_matched` now returns EVERY
    documented domain the phrase matched, so a caller can disclose that its
    banner is one candidate of several rather than presenting it as the only one.
    The resolution itself is unchanged: longest whole-token match still wins, and
    'mortgage bail-in' -> lending and 'health insurance' -> insurance are pinned
    behaviour that depends on it.
    """
    matched = _domain_terms_matched(domain, candidates)
    if matched:
        return matched[0]
    return str(domain or "").strip().lower()


def _domain_terms_matched(
    domain: Optional[str], candidates: Optional[Dict[str, str]] = None
) -> List[str]:
    """Every documented domain a phrase matches as a whole token, longest term first.

    The list :func:`_resolve_domain_key` takes its answer from. More than one
    entry means the phrase carries more than one documented domain and the winner
    was decided by a length tiebreak, which is a fact about the lookup that the
    caller has to publish rather than absorb.
    """
    key = str(domain or "").strip().lower()
    if not key:
        return []
    table = _DOMAIN_ALIASES if candidates is None else candidates
    if key in table:
        # An EXACT match on the whole phrase is not a tiebreak: the caller named a
        # documented term outright, so it is the one answer and nothing else is
        # consulted. Preserved from the original resolution order.
        return [table[key]]
    tokens = name_token_list(key)
    if not tokens:
        return []
    # The canonical keys are matchable terms in their own right ("online hiring
    # platform" -> hiring), which the substring loop never covered because the
    # alias table holds no entry for them.
    terms: Dict[str, str] = {k: k for k in _DOMAIN_HISTORICAL_PRECEDENT}
    terms.update(table)
    # Longest first (by token count, then characters), so 'insurance' beats
    # 'health' in "health insurance" and 'mortgage' beats 'bail' in
    # "mortgage bail-in". A single common word inside a longer phrase is the
    # weakest evidence there is, and it used to win by dict order.
    out: List[str] = []
    for term in sorted(terms, key=lambda t: (-len(name_token_list(t)), -len(t))):
        if tokens_contain(tokens, term):
            canon = terms[term]
            if canon not in out:
                out.append(canon)
    return out


def domain_historical_context(
    domain: str,
    jurisdiction: str = "",
) -> Optional[Dict[str, Any]]:
    """Domain-level documented-precedent banner (spec §3.1).

    Returns a curated, citation-backed context for the regulated domain, or
    None if the domain is unknown (no fabrication: silence beats an
    invented precedent). Citations are real canonical sources only.

    THREE STATES, NEVER TWO. ``None`` is "no documented precedent". A returned
    dict is a precedent, and it says which kind: ``domain_match_is_ambiguous`` is
    False when the phrase resolved to exactly one documented domain, and True when
    it carried SEVERAL, in which case ``domains_matched`` lists them all, the
    banner belongs to the one a character-length tiebreak picked, and
    ``disclosure`` says so in a sentence a reader sees. BGL-F5 2026-09-30: before
    this, 'university hospital clinical trials' returned the ``education``
    precedent led by a facial-recognition citation, with warnings 0 and nothing
    anywhere recording that ``healthcare`` matched too.
    """
    if not domain:
        return None
    key = str(domain).strip().lower()
    key = _DOMAIN_ALIASES.get(key, key)
    matched: List[str] = []
    if key not in _DOMAIN_HISTORICAL_PRECEDENT:
        # Whole-token match on the phrase (e.g. "online hiring platform"), never
        # a substring one. See _resolve_domain_key for the measured before/after.
        matched = [d for d in _domain_terms_matched(key) if d in _DOMAIN_HISTORICAL_PRECEDENT]
        if matched:
            key = matched[0]
    rec = _DOMAIN_HISTORICAL_PRECEDENT.get(key)
    if not rec:
        return None
    out = dict(rec)
    out["domain"] = key
    out["jurisdiction"] = jurisdiction or ""
    out["domains_matched"] = list(matched) if matched else [key]
    ambiguous = len(out["domains_matched"]) > 1
    out["domain_match_is_ambiguous"] = ambiguous
    if ambiguous:
        others = [d for d in out["domains_matched"] if d != key]
        out["disclosure"] = (
            f"{domain!r} matched {len(out['domains_matched'])} documented domains "
            f"({', '.join(out['domains_matched'])}); the precedent below is "
            f"{key!r}'s, chosen because it is the longest whole-token match, and the "
            f"precedent for {', '.join(repr(o) for o in others)} is NOT reported. Which "
            f"one applies is a could-not-check, not a measurement: read the banner as one "
            f"candidate, not as the documented precedent for this audit."
        )
        warnings.warn(f"domain_historical_context: {out['disclosure']}", UserWarning, stacklevel=2)
    return out


# Attribute x domain -> structured historical-discrimination pattern lookup.
#
# Why: the bias engine emits findings of many kinds (selection-rate disparity,
# intersectional, representation, proxy, ...). When the underlying attribute +
# domain combination matches a DOCUMENTED, citation-backed pattern of
# historical discrimination (race in lending => redlining, age in hiring =>
# ADEA, gender in healthcare => clinical-trial under-representation, ...), the
# finding IS a historical pattern even if the finding text uses statistical
# phrasing. Categorising historical patterns on the frontend by regex over
# evidence text under-counts the channel: only findings that happen to mention
# the word "historical" land there. This structured map lets the orchestrator
# attach a `historicalPattern` envelope to every bias finding whose (attribute
# class, domain) combination is documented, so the UI can key the channel off
# a flag instead of free-text matching.
#
# Coverage. Each entry cites the canonical precedent already encoded in
# _DOMAIN_HISTORICAL_PRECEDENT plus, where the attribute deserves a specific
# named pattern (ADEA for age, ADA for disability, Title VII for religion,
# ECOA for national-origin lending, redlining for ZIP), a per-attribute
# citation line. No fabricated cases; silence for unknown combinations.

_ATTR_CLASS_ALIASES: Dict[str, str] = {
    "race": "race",
    "ethnicity": "race",
    "ethnic_group": "race",
    "skin_color": "race",
    "skin_colour": "race",
    "race_ethnicity": "race",
    "gender": "gender",
    "sex": "gender",
    "gender_identity": "gender",
    "age": "age",
    "age_group": "age",
    "age_bracket": "age",
    "national_origin": "national_origin",
    "nationality": "national_origin",
    "country_of_origin": "national_origin",
    "citizenship": "national_origin",
    "disability": "disability",
    "disabled": "disability",
    "disability_status": "disability",
    "impairment": "disability",
    "religion": "religion",
    "religious_affiliation": "religion",
    "faith": "religion",
    # Geographic proxies that ARE the historical-discrimination pattern.
    "zip": "geographic",
    "zip_code": "geographic",
    "zipcode": "geographic",
    "postcode": "geographic",
    "postal_code": "geographic",
    "neighborhood": "geographic",
    "neighbourhood": "geographic",
    "census_tract": "geographic",
    "tract": "geographic",
    "address": "geographic",
}


def _classify_attribute(name: Optional[str]) -> Optional[str]:
    """Map a raw column / attribute name to a canonical attribute class.

    WHOLE-TOKEN, NEVER SUBSTRING. BGL5 2026-09-27. The alias loop used to be
    ``if alias in k``, a raw substring test, and the class it returned is what
    :func:`attribute_historical_pattern` turns into a citation-backed legal
    precedent. Measured before: 'primary_language', 'average_salary',
    'message_count', 'percentage_complete', 'usage_tier', 'package_weight',
    'storage_class' and 'triage_score' all classified as ``age`` and were handed
    ``pattern_id='hiring_age'`` with the citation 'US Age Discrimination in
    Employment Act (ADEA)'; 'contract_type' classified as ``geographic`` and was
    handed 'lending_geographic_redlining' citing Rothstein's 'The Color of Law'.
    All nine now return None, which is what the docstring of the caller promises
    ("silence beats invention"), while 'age_group', 'customerAge',
    'race_ethnicity' and 'census_tract_id' still resolve. ``protected_binning``'s
    module docstring records this exact bug class ('"age" matching
    "primary_language"') as a past incident, and ``vfairness._names`` exists as
    the whole-token fix for it.
    """
    classes = _classify_attribute_classes(name)
    return classes[0] if classes else None


def _classify_attribute_classes(name: Optional[str]) -> List[str]:
    """Every canonical attribute class a name matches whole-token, best first.

    :func:`_classify_attribute` takes its single answer from the front of this
    list. More than one entry means the name carries MORE THAN ONE protected
    class, and the winner was decided by alias length and token order.

    BGL-F5 2026-09-30. That pick was silent, and
    :func:`attribute_historical_pattern` turned it into a citation-backed legal
    precedent for one protected class while DISCARDING the other. Measured before:
    'race_age_group' -> class 'age', pattern 'hiring_age', citing the ADEA, with
    race dropped and warnings 0; reverse the words and 'age_race' -> 'race',
    citing Bertrand & Mullainathan (2004). 'disability_race' -> race,
    'race_disability_status' -> disability (ADA), 'gender_age' and 'age_gender'
    both -> gender (the Amazon recruiting tool). Worst measured case:
    'race_zip_code' -> 'geographic' (the two-token alias 'zip_code' outranks
    'race'), and _ATTR_DOMAIN_PATTERNS['geographic'] has no 'hiring' entry, so
    attribute_historical_pattern('race_zip_code', 'hiring') returned None, i.e. NO
    precedent at all for a column that names race in a hiring audit, which the
    caller's own docstring tells a reader to interpret as "the combination has no
    documented precedent". ``vfairness._names.resolve_attribute_key`` states the
    house rule this was missing: such a None "is a COULD-NOT-CHECK: the caller
    must disclose it rather than reporting the empty lookup as no pattern found".

    An EXACT match on the whole token-joined name short-circuits and is never
    ambiguous: the caller spelled one canonical alias outright.
    """
    tokens = name_token_list(name)
    if not tokens:
        return []
    joined = "_".join(tokens)
    if joined in _ATTR_CLASS_ALIASES:
        return [_ATTR_CLASS_ALIASES[joined]]
    # Longest alias first, so a multi-word alias ('census_tract', 'skin_color')
    # wins over a single-token one it contains.
    out: List[str] = []
    for alias in sorted(_ATTR_CLASS_ALIASES, key=lambda a: -len(name_token_list(a))):
        if tokens_contain(tokens, alias):
            cls = _ATTR_CLASS_ALIASES[alias]
            if cls not in out:
                out.append(cls)
    return out


# Per-(attribute_class, domain) named historical pattern + citation, layered
# on top of the domain-level precedent in _DOMAIN_HISTORICAL_PRECEDENT.
_ATTR_DOMAIN_PATTERNS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "race": {
        "hiring": {
            "pattern_id": "hiring_race",
            "label": "Race in hiring",
            "summary": (
                "Resume-audit and algorithmic-hiring studies document "
                "persistent disparate impact against Black, Hispanic "
                "and other minoritised candidates."
            ),
            "citations": ["Bertrand & Mullainathan (2004)", "Raghavan et al. (2020), FAccT"],
        },
        "lending": {
            "pattern_id": "lending_race_redlining",
            "label": "Race in lending (redlining)",
            "summary": (
                "Credit-allocation models inherit decades of "
                "redlining; ZIP and neighbourhood routinely "
                "reconstruct race even when race is excluded."
            ),
            "citations": ["Rothstein (2017), 'The Color of Law'", "Bartlett et al. (2022), JFE"],
        },
        "healthcare": {
            "pattern_id": "healthcare_race",
            "label": "Race in healthcare risk scoring",
            "summary": (
                "Cost-based health risk algorithms have a landmark "
                "documented case of systematically under-serving "
                "Black patients at equal sickness."
            ),
            "citations": ["Obermeyer et al. (2019), Science"],
        },
        "justice": {
            "pattern_id": "justice_race",
            "label": "Race in recidivism scoring",
            "summary": (
                "Recidivism risk scores have a landmark documented "
                "finding of higher false-positive rates for Black "
                "defendants."
            ),
            "citations": [
                "Angwin et al. (2016), ProPublica (COMPAS)",
                "Chouldechova (2017), Big Data",
            ],
        },
        "insurance": {
            "pattern_id": "insurance_race",
            "label": "Race-correlated insurance pricing",
            "summary": (
                "Insurance underwriting carries geographic and "
                "socioeconomic proxies with documented disparate "
                "impact by race."
            ),
            "citations": ["NAIC / state DOI proxy-discrimination actions"],
        },
        "education": {
            "pattern_id": "education_race",
            "label": "Race in admissions",
            "summary": (
                "Admissions models inherit standardised-test and "
                "legacy-preference disparities that disadvantage "
                "historically excluded racial groups."
            ),
            "citations": ["Buolamwini & Gebru (2018), Gender Shades"],
        },
    },
    "gender": {
        "hiring": {
            "pattern_id": "hiring_gender",
            "label": "Gender in hiring (pay gap, screening bias)",
            "summary": (
                "Hiring algorithms have documented disparate impact "
                "against women (Amazon's scrapped recruiter, gendered "
                "language penalties)."
            ),
            "citations": ["Dastin (2018), Reuters (Amazon)", "Raghavan et al. (2020), FAccT"],
        },
        "lending": {
            "pattern_id": "lending_gender",
            "label": "Gender in credit access",
            "summary": (
                "Credit-allocation models have documented gender "
                "disparities in approval and pricing even controlling "
                "for risk."
            ),
            "citations": ["ECOA / Reg-B enforcement record"],
        },
        "healthcare": {
            "pattern_id": "healthcare_gender",
            "label": "Gender in clinical decision-making",
            "summary": (
                "Clinical-trial under-representation of women and "
                "gender-coded diagnostic patterns create systematic "
                "diagnostic gaps."
            ),
            "citations": ["FDA gender-in-clinical-trials guidance"],
        },
        "insurance": {
            "pattern_id": "insurance_gender",
            "label": "Gender in insurance pricing",
            "summary": (
                "Gender has documented use as a pricing proxy; EU "
                "Test-Achats (2011) struck down gender-based insurance "
                "pricing."
            ),
            "citations": ["CJEU Test-Achats (2011)"],
        },
        "education": {
            "pattern_id": "education_gender",
            "label": "Gender in education / admissions",
            "summary": (
                "STEM admissions and recommendation models carry documented gender disparities."
            ),
            "citations": ["EU AI Act Annex III(3)"],
        },
    },
    "age": {
        "hiring": {
            "pattern_id": "hiring_age",
            "label": "Age discrimination in hiring (ADEA)",
            "summary": (
                "Resume-audit studies and algorithmic-screening "
                "audits document disparate impact against older "
                "applicants; the ADEA (US) and EU equal-treatment "
                "directive prohibit age discrimination."
            ),
            "citations": [
                "US Age Discrimination in Employment Act (ADEA)",
                "EU Directive 2000/78/EC",
            ],
        },
        "lending": {
            "pattern_id": "lending_age",
            "label": "Age in credit",
            "summary": (
                "ECOA explicitly lists age as a prohibited basis in "
                "credit; age proxies (graduation year, employment "
                "tenure) recreate the same disparity."
            ),
            "citations": ["US ECOA / Regulation B"],
        },
        "insurance": {
            "pattern_id": "insurance_age",
            "label": "Age in insurance pricing",
            "summary": (
                "Age is a documented pricing proxy with disparate "
                "impact on elderly cohorts in non-life lines."
            ),
            "citations": ["EU AI Act Annex III(5)(c)"],
        },
    },
    "national_origin": {
        "hiring": {
            "pattern_id": "hiring_national_origin",
            "label": "National-origin discrimination (Title VII)",
            "summary": (
                "Resume-audit studies document disparate impact by "
                "name-origin and accent; Title VII prohibits "
                "national-origin discrimination in employment."
            ),
            "citations": ["US EEOC Title VII", "Bertrand & Mullainathan (2004)"],
        },
        "lending": {
            "pattern_id": "lending_national_origin",
            "label": "National origin in credit (ECOA)",
            "summary": (
                "National origin is a prohibited basis under ECOA; "
                "language and name proxies reconstruct it."
            ),
            "citations": ["US ECOA / Regulation B"],
        },
    },
    "disability": {
        "hiring": {
            "pattern_id": "hiring_disability",
            "label": "Disability discrimination in hiring (ADA)",
            "summary": (
                "The ADA (US) and EU Directive 2000/78/EC prohibit "
                "disability discrimination; algorithmic-screening "
                "audits document disparate impact."
            ),
            "citations": ["US Americans with Disabilities Act (ADA)", "EU Directive 2000/78/EC"],
        },
        "healthcare": {
            "pattern_id": "healthcare_disability",
            "label": "Disability in healthcare access",
            "summary": (
                "Triage and access algorithms have documented "
                "disparate impact against patients with disabilities."
            ),
            "citations": ["US Section 1557 ACA"],
        },
        "insurance": {
            "pattern_id": "insurance_disability",
            "label": "Disability in insurance",
            "summary": (
                "Disability has documented use as an underwriting "
                "proxy; ADA Title III and EU Directive constrain it."
            ),
            "citations": ["US ADA Title III"],
        },
    },
    "religion": {
        "hiring": {
            "pattern_id": "hiring_religion",
            "label": "Religion in hiring (Title VII)",
            "summary": (
                "Resume-audit studies document disparate impact by "
                "religiously-coded names; Title VII prohibits "
                "religion-based hiring discrimination."
            ),
            "citations": ["US EEOC Title VII"],
        },
        "lending": {
            "pattern_id": "lending_religion",
            "label": "Religion in credit (ECOA)",
            "summary": ("Religion is a prohibited basis under ECOA."),
            "citations": ["US ECOA / Regulation B"],
        },
    },
    # Geographic proxies are the historical pattern itself in lending /
    # insurance / healthcare access (redlining). For hiring, geographic
    # features more typically proxy for socioeconomic status than a single
    # named precedent, so it's omitted there to avoid over-flagging.
    "geographic": {
        "lending": {
            "pattern_id": "lending_geographic_redlining",
            "label": "Geographic redlining (lending)",
            "summary": (
                "ZIP / neighbourhood reconstruct race in credit "
                "decisions; redlining is the named historical "
                "pattern."
            ),
            "citations": ["Rothstein (2017), 'The Color of Law'", "Bartlett et al. (2022), JFE"],
        },
        "insurance": {
            "pattern_id": "insurance_geographic_redlining",
            "label": "Geographic redlining (insurance)",
            "summary": (
                "Geographic underwriting proxies have documented "
                "disparate impact; named insurance redlining."
            ),
            "citations": ["NAIC proxy-discrimination actions"],
        },
        "healthcare": {
            "pattern_id": "healthcare_geographic_access",
            "label": "Geographic disparity in healthcare access",
            "summary": (
                "Service-area and ZIP features encode well-documented "
                "racial and socioeconomic access gaps."
            ),
            "citations": ["US Section 1557 ACA"],
        },
    },
}


def attribute_historical_pattern(
    attribute: Optional[str],
    domain: Optional[str],
    jurisdiction: str = "",
) -> Optional[Dict[str, Any]]:
    """Resolve (attribute, domain) to a structured historical pattern.

    Returns a dict with `pattern_id`, `label`, `summary`, `citations`,
    `attribute_class`, `domain`, `jurisdiction`, or None when the combination
    has no documented precedent (silence beats invention).

    THREE STATES, NEVER TWO. ``None`` means "no documented precedent". A dict with
    a ``pattern_id`` is a precedent. A dict whose ``pattern_id`` is None and whose
    ``measurement_status`` is ``'could_not_check'`` is the third state: the
    attribute name carried MORE THAN ONE protected class (named in
    ``attribute_classes_matched``), so which legal analysis applies could not be
    determined, no precedent and no citation is reported, and ``not_matched_reason``
    says why where a reader looks. BGL-F5 2026-09-30, see
    :func:`_classify_attribute_classes` for the measured before-state.
    """
    classes = _classify_attribute_classes(attribute)
    if len(classes) > 1 and domain:
        # A COULD-NOT-CHECK ENVELOPE, NOT A PICK AND NOT SILENCE. Choosing one of
        # two protected classes by alias length drops the other from a legal
        # analysis, and in the 'race_zip_code' case the pick produced no precedent
        # at all, which the docstring above tells a reader to read as "no
        # documented precedent exists". Every key the success shape has is present
        # so a consumer that indexes them cannot break; pattern_id is None and
        # citations is empty, so no precedent and no legal citation is asserted.
        reason = (
            f"{attribute!r} carries {len(classes)} protected attribute classes "
            f"({', '.join(classes)}), so which documented historical pattern applies "
            f"could not be determined. Reporting a could-not-check rather than the "
            f"{classes[0]!r} precedent, which an alias-length tiebreak would have picked "
            f"while discarding {', '.join(repr(c) for c in classes[1:])}. Split the column, "
            f"or pass one class explicitly."
        )
        warnings.warn(f"attribute_historical_pattern: {reason}", UserWarning, stacklevel=2)
        return {
            "pattern_id": None,
            "label": "Ambiguous attribute: no single documented pattern",
            "summary": reason,
            "citations": [],
            "attribute_class": None,
            "attribute_classes_matched": list(classes),
            "measurement_status": "could_not_check",
            "not_matched_reason": reason,
            "domain": str(domain).strip().lower(),
            "jurisdiction": jurisdiction or "",
        }
    cls = classes[0] if classes else None
    if not cls or not domain:
        return None
    dkey = str(domain).strip().lower()
    dkey = _DOMAIN_ALIASES.get(dkey, dkey)
    if dkey not in _ATTR_DOMAIN_PATTERNS.get(cls, {}):
        # Whole-token, longest match first, and only onto a domain this attribute
        # class actually has a documented pattern for. See _resolve_domain_key.
        _usable = {
            alias: canon
            for alias, canon in _DOMAIN_ALIASES.items()
            if canon in _ATTR_DOMAIN_PATTERNS.get(cls, {})
        }
        _matched_domains = [
            d
            for d in _domain_terms_matched(dkey, candidates=_usable)
            if d in _ATTR_DOMAIN_PATTERNS.get(cls, {})
        ]
        if _matched_domains:
            dkey = _matched_domains[0]
    else:
        _matched_domains = [dkey]
    rec = _ATTR_DOMAIN_PATTERNS.get(cls, {}).get(dkey)
    if not rec:
        return None
    out = dict(rec)
    out["attribute_class"] = cls
    out["domain"] = dkey
    out["jurisdiction"] = jurisdiction or ""
    # The three states on the surface, not inferred from a None return: this one
    # IS a measured precedent, and it says which attribute class and which domain
    # it was measured for. BGL-F5 2026-09-30.
    out["attribute_classes_matched"] = list(classes)
    out["measurement_status"] = "measured"
    out["domains_matched"] = list(_matched_domains) or [dkey]
    out["domain_match_is_ambiguous"] = len(out["domains_matched"]) > 1
    if out["domain_match_is_ambiguous"]:
        _others = [d for d in out["domains_matched"] if d != dkey]
        out["disclosure"] = (
            f"{domain!r} matched {len(out['domains_matched'])} documented domains for "
            f"{cls!r} ({', '.join(out['domains_matched'])}); the pattern below is "
            f"{dkey!r}'s, chosen because it is the longest whole-token match, and the "
            f"pattern for {', '.join(repr(o) for o in _others)} is NOT reported."
        )
        warnings.warn(
            f"attribute_historical_pattern: {out['disclosure']}", UserWarning, stacklevel=2
        )
    return out
