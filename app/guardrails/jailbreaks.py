
import re


# ---------------------------------------------------------
# 1. JAILBREAK DETECTION
# ---------------------------------------------------------

JAILBREAK_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above|earlier|system)\s+instructions\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions\b",
    r"\b(reveal|show|print|expose|repeat)\b.{0,60}\b(system prompt|hidden prompt|internal instructions)\b",
    r"\b(bypass|disable|override|remove|ignore)\b.{0,60}\b(guardrails|safety rules|restrictions|instructions)\b",
    r"\bact as\b.{0,40}\b(unrestricted|unfiltered|uncensored|jailbroken)\b",
    r"\b(developer mode|DAN mode|jailbreak mode)\b",
    r"\bpretend\b.{0,50}\b(no rules|no restrictions|without safeguards)\b",
    r"\b(do not follow|stop following)\b.{0,40}\b(system|developer|safety)\s+(instructions|rules)\b",
    r"\b(output|print|reveal|leak)\b.{0,40}\b(api key|secret key|credentials)\b",
]


# ---------------------------------------------------------
# 2. DOCUMENT-WIDE HR POLICY SCOPE DETECTION
# ---------------------------------------------------------

HR_POLICY_PATTERNS = [

        # A. Organizational identity, mission, vision and values

    r"\b(?:policy manual|employee handbook|staff handbook|hr policy|human resources policy)\b",

    r"\bopengov(?:\s+foundation)?\b",

    r"\b(?:mission|vision|values?|core values|organizational values|"
    r"company values|foundation values|guiding principles)\b",

    r"\b(?:organizational purpose|organization's purpose|"
    r"organization purpose|purpose of the organization|"
    r"purpose of this organization|purpose of the company|"
    r"company purpose|organizational objectives|"
    r"organization's objectives|strategic objectives|"
    r"organizational goals|organization's goals|"
    r"long[- ]term goals|organizational philosophy)\b",

    r"\b(?:what does the organization stand for|"
    r"what does this organization stand for|"
    r"what does the company stand for|"
    r"what is the organization trying to achieve|"
    r"what is the company trying to achieve|"
    r"what is the organization's purpose|"
    r"why does the organization exist|"
    r"why was the organization established|"
    r"what is the organization about)\b",
    r"\b(?:organization|organisation|company|business|foundation|employer)\b",
    r"\b(?:vision|mission|values|purpose|objectives|goals)\b",
    r"\b(?:staff|team|management|leadership|directors?|advisers?)\b",

    # B. Employees, staff and organizational structure
    r"\b(?:hr|human resources|employees?|staff|team members?|applicants?|supervisors?|coworkers?|colleagues?)\b",
    r"\b(?:board of directors|board of advisers|board of advisors|advisory board)\b",
    r"\b(?:executive director|deputy executive director|chief of staff|director of product|senior engineer|co-founder|vice chair|chairperson|treasurer|secretary)\b",

    # Names appearing in the staff, board and adviser sections
    r"\b(?:seamus kraft|patrick bateman|aaron ogle|seth etter|tanner doshier|kunal kothari|phaedra chrousos|laurent crenshaw|rich hirshberg|michelle lee|aaron snow|scout addis|karien bezuidenhout|anne washington|aaron bartnick|brandon andrews|jo[- ]marie st\.?\s*martin|peter karman|janine gianfredi|bob sofman)\b",

    # C. Equal employment, discrimination and accommodation
    r"\b(?:equal employment|equal employment opportunity|non[- ]discrimination|discrimination|discriminatory|protected classes?|protected categories|reasonable accommodation|accommodations?)\b",
    r"\b(?:race|religion|national origin|ancestry|genetic information|sexual orientation|gender identity|gender expression|parental status|marital status|political affiliation|veteran status|military service|pregnancy|childbirth|medical condition|neuro\(a\)typicality|socioeconomic status)\b",
    r"\b(?:physical disability|mental disability|mental illness|physical appearance|body size|transgender status|sex stereotyping)\b",

    # D. Harassment, bullying, violence and workplace safety
    r"\b(?:harass\w*|bully\w*|bullying|workplace violence|hostile environment|retaliat\w*|unwelcome conduct)\b",
    r"\b(?:slurs?|epithets?|derogatory comments?|offensive remarks?|unwelcome jokes?|teasing|belittling|patronizing|marginaliz\w*)\b",
    r"\b(?:affirmative consent|unwanted touching|incorrect pronouns|gender pronouns|protected class harassment)\b",
    r"\b(?:harassment complaint|workplace complaint|reporting violations|report a violation|complaint investigation|disciplinary action|termination for misconduct)\b",

    # E. Reporting and resolution of violations
    r"\b(?:reporting and resolution|report a complaint|file a complaint|witnessed a violation|investigat\w*|confidential investigation|good faith report|adverse action|retaliation for reporting|violation of this policy)\b",
    r"\b(?:up to and including discharge|management complaint|report to management|escalate a complaint)\b",

    # F. Drug and alcohol policy
    r"\b(?:drugs?|alcohol|alchohol|controlled substances?|unauthorized substances?|illegal substances?|substance use|substance abuse|drug abuse|alcohol abuse|drug and alcohol treatment|treatment services)\b",
    r"\b(?:reporting to work under the influence|working under the influence|drug treatment|alcohol treatment|substance treatment)\b",

    # G. Conflict of interest
    r"\b(?:conflict of interest|conflicting interests?|private gain|personal gain|personal interests?|personal favors?|private benefits?|undue benefits?)\b",
    r"\b(?:family members.*(?:benefit|favor)|perceived conflict|perception of a conflict|business entities|corporate interests|allowable conflict|declare a conflict|disclose a conflict)\b",

    # H. Professional conduct, ethics and legal standards
    r"\b(?:professional conduct|code of conduct|professional ethics|workplace ethics|ethical standards|legal standards|integrity of employees|letter and spirit of the law)\b",
    r"\b(?:applicable laws|ethical behavior|ethical behaviour|professional standards|legal compliance)\b",

    # I. Contact with the media and organizational communications
    r"\b(?:contact with the media|media inquiries|press inquiries|press contact|media requests|speak to the media|media relations)\b",
    r"\b(?:organizational communications|company communications|speak on behalf of the organization|personal views|personal capacity|free speech|authorized spokesperson)\b",

    # J. Payroll, compensation and benefits
    r"\b(?:benefits?|compensation|payroll|payday|pay dates?|salary|salaries|wages?|direct deposit|payroll deduction|twice monthly|15th of the month|last day of the month)\b",
    r"\b(?:cost of living increase|cost[- ]of[- ]living adjustment|merit increase|merit raise|raises?|bonuses?|annual increase|salary review|position change|increased responsibilities)\b",
    r"\b(?:board approval for raises|performance assessment|performance review|employee evaluation|review of performance|performance methodology)\b",

    # K. Medical, dental, vision and pharmacy coverage
    r"\b(?:medical coverage|dental coverage|vision coverage|health insurance|employer[- ]sponsored health insurance|pharmacy benefits?|insurance plan|benefit enrollment|open enrollment)\b",
    r"\b(?:insurance premium|premium cost|employee premium|employer contribution|payroll deduction|optional enrollment|onboarding benefits|health plan coverage)\b",

    # L. Reimbursable expenses and business travel
    r"\b(?:reimbursable expenses?|expense reimbursement|expense claims?|reimbursement request|work[- ]related expenses?|pre[- ]approved expenses?)\b",
    r"\b(?:expense approval|receipt submission|expense receipts?|business travel|travel expenses?|economy class|economy travel|hotel expenses?|conference expenses?|business meeting meals?)\b",
    r"\b(?:office supplies|software subscriptions?|organization[- ]specific subscriptions?|event hosting|event expenses?|food and drink expenses?|travel in town|travel out of town)\b",
    r"\b(?:one thousand dollars|\$1,000|\$1000|under \$1,000|over \$1,000|board approval for expenses|executive director approval|expense classification|expense categories)\b",

    # M. Professional development
    r"\b(?:professional development|employee development|career development|professional conferences?|relevant classes|tuition|training courses?|development opportunities|learning opportunities)\b",
    r"\b(?:professional development budget|operating budget|development funding|development activities|development approval|development expenditure|tuition reimbursement)\b",

    # N. Schedule, working hours, availability and remote work
    r"\b(?:work schedule|working hours|work hours|hours per week|40 hours|workweek|flexible schedule|flexible working|remote work|work from home|work remotely|local timezone|local time zone)\b",
    r"\b(?:11\s*(?:am|a\.m\.)\s*[-–to]+\s*4\s*(?:pm|p\.m\.)|9\s*(?:am|a\.m\.)\s*[-–to]+\s*5\s*(?:pm|p\.m\.))\b",
    r"\b(?:required meeting|meeting attendance|meeting participant|work calendar|calendar availability|team availability|accessible by phone|slack and email)\b",

    # O. Vacation, holidays and other leave
    r"\b(?:vacation|paid vacation|vacation entitlement|vacation days|paid time off|pto|holidays?|federal holidays?|thanksgiving|christmas|new year|year[- ]end closure)\b",
    r"\b(?:leave policy|leave entitlement|family leave|medical leave|personal leave|maternity leave|paternity leave|parental leave|bereavement leave|sick leave|sick days?)\b",
    r"\b(?:family and medical leave act|fmla|16 weeks|two weeks|10 business days|extended leave|time off|schedule time off|leave approval)\b",
    r"\b(?:notify your supervisor|written leave request|advance notice|five business days|5 business days|leave calendar|leave planning)\b",

    # P. Sick days and illness
    r"\b(?:sick day|sick days|feeling sick|illness|recovering from illness|work from home while sick|notify the team|contagious illness|avoid spreading illness)\b",

    # Q. Email accounts, document sharing and password sharing
    r"\b(?:google suite|organizational email|organization email|work email|work[- ]related correspondence|personal accounts?|document sharing|shared storage)\b",
    r"\b(?:password sharing|share passwords?|shared passwords?|password manager|shared password manager|passwords via email|insecure password sharing)\b",

    # R. Calendars and availability
    r"\b(?:google calendar|work calendar|calendar account|calendar availability|record vacation time|calendar appointments?|schedule meetings?|meeting scheduling)\b",

    # S. Regular meetings and daily standups
    r"\b(?:regular meetings?|daily stand[- ]?ups?|stand[- ]?ups?|team meetings?|meeting participation|full participation|geographic location|meeting duration|15 minutes|daily coordination)\b",
    r"\b(?:previous day|yesterday's work|today's work|daily blockers?|work obstacles?|project blockers?)\b",

    # T. Book clubs, family meals and informal rituals
    r"\b(?:book clubs?|family meals?|informal rituals?|informal gatherings?|employee gatherings?|employee groups?|sponsor a gathering|organizational health|team building)\b",

    # U. Tools, services and platforms
    r"\b(?:tools and services|tools and platforms|work platforms|organizational tools|software tools|trial and adoption|tool adoption|tool siloing|operational alignment|new platforms)\b",

    # V. Recommended reading
    r"\b(?:recommended reading|remote office not required|how to collaborate remotely|beware the dragons|managing remote workers|life as a remote worker|dealing with excuses|the lean startup|don't make me think|do not make me think)\b",

    # W. General policy-related wording and contextual terms
    r"\b(?:according to the manual|according to the policy|under this policy|under the manual|employee handbook|company handbook|organization policy|organizational policy|policy requirements|policy rules)\b",

     # Organisational mission, vision and values
    r"\b(?:mission|vision|values?|core values|guiding principles)\b",

    # Media communication
    r"\b(?:media|press|spokesperson|public statement|public relations)\b",

    # Expenses and reimbursement
    r"\b(?:expenses?|reimburs\w*|receipts?|business travel|travel expenses?)\b",

    # Remote working
    r"\b(?:remote work(?:ing)?|working remotely|work from home|remote employees?)\b",

    # Passwords and credential sharing
    r"\b(?:passwords?|password manager|sharing passwords?|password sharing|credential sharing)\b",
]



# ---------------------------------------------------------
# 3. CONTEXTUAL FOLLOW-UP QUESTIONS
# ---------------------------------------------------------

FOLLOW_UP_PATTERNS = [
    r"^(?:what about|how about|what if)\b",
    r"^(?:does that|does this|is that|is this)\b",
    r"^(?:what happens if|what happens next)\b",
    r"^(?:can they do that|can i do that|would that be allowed)\b",
]


# ---------------------------------------------------------
# 4. JAILBREAK VALIDATION
# ---------------------------------------------------------

def is_jailbreak_attempt(text: str) -> bool:
    """Detect common direct or embedded jailbreak phrases."""

    if not isinstance(text, str):
        return False

    normalized = re.sub(r"\s+", " ", text).strip().lower()

    return any(
        re.search(pattern, normalized)
        for pattern in JAILBREAK_PATTERNS
    )


def contains_suspicious_document_instructions(context: str) -> bool:
    """Flag retrieved text containing instruction-override patterns."""

    return is_jailbreak_attempt(context)


# ---------------------------------------------------------
# 5. HR DOCUMENT SCOPE VALIDATION
# ---------------------------------------------------------

def is_hr_policy_question(text: str) -> bool:
    """Check whether a question relates to any topic in the policy manual."""

    if not isinstance(text, str) or not text.strip():
        return False

    normalized = re.sub(r"\s+", " ", text).strip().lower()

    return any(
        re.search(pattern, normalized)
        for pattern in HR_POLICY_PATTERNS + FOLLOW_UP_PATTERNS
    )
