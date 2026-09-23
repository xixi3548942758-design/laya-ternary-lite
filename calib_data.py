"""校准语料：为 GPTQ 风格的三值校准构造有代表性的 (state, questions) 输入。

laya 是「状态 + 类型化问题」的决策模型，输入形态固定为
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 ... [SEP] state [SEP]
所以校准语料要覆盖真实业务里会出现的 state（邮件、工单、对话、JSON、评论文本）
和三种题型（choice / score / noul）。

没有真实业务数据是这套方案最大的近似，样本刻意做得杂，宁可覆盖宽一点。
"""
import random

# --------------------------------------------------------------------------- state 语料
_EMAILS = [
    ("user@acme.com", "Duplicate charge on invoice #4411",
     "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."),
    ("dana.lee@corp.io", "Cannot log in after password reset",
     "I reset my password twice but the login page still says invalid credentials. This blocks my whole team."),
    ("ops@retailco.com", "Bulk pricing for 500 seats",
     "We are evaluating your enterprise tier for 500 seats. What volume discount can you offer, and what is the contract term?"),
    ("mike@startup.dev", "API returning 500 on /v2/orders",
     "Every call to /v2/orders fails with a 500 since 14:20 UTC. Our checkout is down and we are losing sales."),
    ("billing@partner.net", "Invoice address is wrong",
     "The invoice lists our old address. Please correct it and reissue for accounting before month end."),
    ("angry.customer@gmail.com", "Third time asking for a refund",
     "This is the third time I have asked. If nobody responds by Friday I am disputing the charge with my bank."),
    ("hr@bigfirm.com", "Seat count mismatch after upgrade",
     "We upgraded to 200 seats but the admin console shows 150 active. Please reconcile the subscription."),
    ("devrel@community.org", "Feature request: webhook retries",
     "Please add configurable retry policy for webhooks. Current behaviour drops events after one failure."),
    ("cfo@manufacture.cn", "Request for SOC 2 report",
     "Our security review needs your SOC 2 Type II report and a completed vendor questionnaire."),
    ("student@university.edu", "Discount for academic use",
     "I am a graduate student. Is there an academic license or free tier for coursework?"),
]

_TICKETS = [
    "Checkout latency spiked to 4.2s p95 after the 13:00 deploy. Rolled back, latency normal. Root cause under investigation.",
    "Customer reports the export button produces an empty CSV when the date range spans a DST boundary.",
    "Search results omit items uploaded in the last 10 minutes; index lag appears to be around 15 minutes.",
    "Mobile app crashes on launch for Android 14 users after the 3.8.1 update. Crash rate 12% of sessions.",
    "Two-factor authentication codes arrive 5-10 minutes late, making login effectively impossible.",
    "The nightly billing job failed with a deadlock on the subscriptions table. Retry succeeded, no data loss.",
    "Users in the EU report the cookie banner reappears on every page load after consent.",
    "Webhook deliveries to a customer endpoint time out at 30s; their endpoint responds in under 200ms.",
]

_CONVOS = [
    [{"role": "user", "text": "My order has not shipped yet."},
     {"role": "agent", "text": "Let me check the warehouse status for you."},
     {"role": "user", "text": "It has been nine days. I need it by Saturday or I want a refund."}],
    [{"role": "user", "text": "Can I upgrade mid-cycle?"},
     {"role": "agent", "text": "Yes, upgrades are prorated automatically."},
     {"role": "user", "text": "Great, please proceed and send the new invoice."}],
    [{"role": "user", "text": "The dashboard is showing stale numbers."},
     {"role": "agent", "text": "We are aware of a caching issue and are rolling out a fix."},
     {"role": "user", "text": "This is affecting our monthly reporting. How long until it is fixed?"}],
]

_JSONS = [
    {"order_id": "A-88213", "status": "delivered", "amount": 249.99, "currency": "USD",
     "items": 3, "return_requested": False, "days_since_delivery": 4},
    {"account": "acct_9931", "plan": "pro", "seats": 45, "mrr": 1350.0, "past_due": True,
     "last_payment_failed": "2026-09-11", "dunning_attempts": 2},
    {"ticket": "T-40122", "priority": "P1", "component": "payments", "age_hours": 3.5,
     "assignee": None, "customer_tier": "enterprise", "sla_breach": False},
]

_REVIEWS = [
    "The product does exactly what it promises and support replied within an hour. Very happy.",
    "Terrible experience. The import tool corrupted my data and nobody has responded in four days.",
    "Decent value for the price, though the reporting module is slow on large datasets.",
    "Switched from a competitor and the migration was painless. Documentation is excellent.",
    "It works, but the mobile app feels unfinished and crashes occasionally.",
]

_DOCS = [
    "Refunds are processed to the original payment method within five to ten business days. "
    "If the original method is unavailable, account credit is issued instead.",
    "The rate limit is 600 requests per minute per API key. Exceeding it returns HTTP 429 with a Retry-After header.",
    "Enterprise plans include a 99.95% uptime SLA, SSO, audit logs, and a dedicated support channel.",
    "Data is encrypted at rest with AES-256 and in transit with TLS 1.3. Keys rotate every 90 days.",
    "Cancellation takes effect at the end of the current billing period. No partial refunds are issued.",
]

_PROMPTS = [
    ("Which team should own this?", "route the request to the correct internal team"),
    ("How urgent is this?", "judge the time pressure behind the request"),
    ("Is the customer likely to churn?", "detect explicit or implied intent to leave"),
    ("Does this contain a refund request?", "detect an explicit request to return money"),
    ("What is the sentiment?", "classify the emotional tone of the text"),
    ("Which category does this belong to?", "assign the text to a topical category"),
    ("Is this safe to publish?", "detect policy-violating or unsafe content"),
    ("Did the agent resolve the issue?", "judge whether the support reply solved the problem"),
    ("Which language is this written in?", "identify the primary language"),
    ("Is this a bug report or a feature request?", "distinguish defects from enhancements"),
]

_CHOICES = [
    ("Which team should own this?", {"billing": "invoices, payments, refunds",
                                     "technical": "bugs, outages, system errors",
                                     "sales": "pricing, contracts, new business",
                                     "other": "everything else"}),
    ("Which category does this belong to?", {"account": "login, seats, permissions",
                                             "payment": "charges, invoices, dunning",
                                             "product": "features, bugs, usability",
                                             "other": "uncategorised"}),
    ("What is the sentiment?", {"positive": "satisfied, appreciative",
                                "neutral": "factual, no strong feeling",
                                "negative": "frustrated, angry, disappointed"}),
    ("What is the priority?", {"low": "no time pressure", "medium": "soon but not blocking",
                               "high": "blocking work", "critical": "revenue or safety impact"}),
]

_SCORES = [
    ("How urgent is this request?", ["not urgent", "soon", "critical deadline or blocking issue"]),
    ("How would you rate the response quality?", ["poor", "acceptable", "good", "excellent"]),
    ("How severe is the impact?", ["cosmetic", "minor friction", "major disruption", "business stopped"]),
    ("How confident are you in this classification?", ["guess", "weak", "fairly sure", "certain"]),
]

_NOULS = [
    "Does the customer threaten to cancel or leave?",
    "Does the customer explicitly request a refund?",
    "Is a deadline mentioned?",
    "Does the text contain personal data?",
    "Has the issue already been resolved?",
    "Is the customer asking about pricing?",
    "Does the message reference a competitor?",
]


# --------------------------------------------------------------------------- 组装
def _all_states(rng):
    out = []
    for f, s, b in _EMAILS:
        out.append({"from": f, "subject": s, "body": b})
        out.append("%s\n\n%s" % (s, b))
    out += list(_TICKETS)
    out += [{"conversation": c} for c in _CONVOS]
    out += list(_JSONS)
    out += list(_REVIEWS)
    out += list(_DOCS)
    rng.shuffle(out)
    return out


def _question(rng, qtype=None):
    qtype = qtype or rng.choice(["choice", "score", "noul"])
    if qtype == "choice":
        ins, crit = rng.choice(_CHOICES)
        return {"t": "choice", "ins": ins, "crit": crit}
    if qtype == "score":
        ins, crit = rng.choice(_SCORES)
        return {"t": "score", "ins": ins, "crit": crit}
    return {"t": "noul", "ins": rng.choice(_NOULS), "crit": {"false": None, "true": None}}


def build_calibration_set(n_samples=256, seed=1234):
    """返回 n_samples 条 (state, [question, ...])，题量 1~4 条以贴合真实调用。"""
    rng = random.Random(seed)
    states = _all_states(rng)
    samples = []
    for i in range(n_samples):
        state = states[i % len(states)]
        nq = rng.choice([1, 1, 2, 2, 3, 4])
        qs = [_question(rng) for _ in range(nq)]
        samples.append((state, qs))
    return samples


if __name__ == "__main__":
    import collections
    import json
    s = build_calibration_set()
    c = collections.Counter(q["t"] for _, qs in s for q in qs)
    print("样本数:", len(s), " 题目分布:", dict(c))
    print("state 类型分布:", collections.Counter(
        "json" if isinstance(st, dict) and "order_id" in st or isinstance(st, dict) and "account" in st
        else "convo" if isinstance(st, dict) and "conversation" in st
        else "email" if isinstance(st, dict) else "text" for st, _ in s))
    st, qs = s[0]
    print("\n示例 state:", json.dumps(st, ensure_ascii=False)[:200])
    print("示例 questions:", json.dumps(qs, ensure_ascii=False)[:300])
