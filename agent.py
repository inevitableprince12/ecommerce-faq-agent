from sentence_transformers import SentenceTransformer
import chromadb
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# DB
client = chromadb.Client()
collection = client.get_or_create_collection(name="ecommerce_kb")

# 👉 paste your FULL knowledge_base here
knowledge_base = [
    {
        "id": "doc_001",
        "topic": "Order Placement Process",
        "text": (
            "Placing an order on our platform is a straightforward, step-by-step process designed for ease and speed.\n\n"
            "How to place an order:\n"
            "1. Browse or search for the product you want.\n"
            "2. Select the correct size, color, or variant, then click 'Add to Cart'.\n"
            "3. Review your cart — update quantities or remove items as needed.\n"
            "4. Click 'Proceed to Checkout'.\n"
            "5. Enter or confirm your shipping address.\n"
            "6. Choose a payment method and complete payment.\n"
            "7. Review the order summary and click 'Place Order'.\n\n"
            "Once your order is confirmed, you will receive an email confirmation with your Order ID within a few minutes. "
            "Guest checkout is available, but creating an account lets you track orders, save addresses, and manage returns more easily. "
            "All orders are subject to product availability. If an item becomes out of stock after you place your order, our support team will notify you promptly and offer alternatives or a full refund. "
            "Minimum order value requirements may apply during certain promotions."
        )
    },
    {
        "id": "doc_002",
        "topic": "Payment Methods",
        "text": (
            "We support a wide range of secure payment methods to make checkout convenient for every customer.\n\n"
            "Accepted payment methods:\n"
            "- Credit and Debit Cards: Visa, MasterCard, American Express, RuPay\n"
            "- UPI: Google Pay, PhonePe, Paytm UPI, BHIM\n"
            "- Net Banking: All major Indian banks supported\n"
            "- Digital Wallets: Paytm Wallet, Amazon Pay, Mobikwik\n"
            "- Buy Now Pay Later (BNPL): Available via partners like Simpl and LazyPay\n"
            "- EMI: No-cost EMI on orders above ₹3,000 with select bank cards\n"
            "- Cash on Delivery (COD): Available on orders up to ₹10,000 in eligible pin codes\n\n"
            "All transactions are encrypted using 256-bit SSL technology. We do not store your full card details. "
            "Payments are processed in real time and you will receive instant confirmation. "
            "If a payment fails, no amount is debited — any temporary hold is released within 2–5 business days by your bank. "
            "International cards are accepted, but additional foreign transaction fees from your bank may apply."
        )
    },
    {
        "id": "doc_003",
        "topic": "Shipping & Delivery Timelines",
        "text": (
            "Shipping timelines vary based on the delivery type selected and your location.\n\n"
            "Delivery options and estimated timelines:\n"
            "- Standard Delivery: 5–7 business days (free on orders above ₹499)\n"
            "- Express Delivery: 2–3 business days (₹99 flat fee)\n"
            "- Same-Day Delivery: Available in select metro cities for orders placed before 11 AM (₹149 fee)\n"
            "- Next-Day Delivery: Available in Tier-1 cities (₹99 fee)\n\n"
            "Orders are typically processed and dispatched within 24–48 hours of payment confirmation, excluding Sundays and public holidays. "
            "Delivery timelines start from the dispatch date, not the order date. "
            "Remote and rural pin codes may have extended delivery windows of up to 10 business days. "
            "During peak sale events (e.g., festive season, flash sales), processing times may be slightly longer. "
            "Free shipping thresholds and fees are subject to change — always check the checkout page for the most current rates. "
            "All shipments are insured against transit damage."
        )
    },
    {
        "id": "doc_004",
        "topic": "Order Tracking",
        "text": (
            "Once your order is dispatched, you can track it in real time using multiple methods.\n\n"
            "How to track your order:\n"
            "- Log in to your account and go to 'My Orders' → select the order → click 'Track Order'.\n"
            "- Use the tracking link sent to your registered email or SMS after dispatch.\n"
            "- Enter your Order ID and registered email/phone on our Track Order page (no login required).\n\n"
            "Tracking information is updated by our logistics partner and typically reflects shipment status within 12–24 hours of dispatch. "
            "Status milestones you will see:\n"
            "1. Order Confirmed\n"
            "2. Processing\n"
            "3. Dispatched\n"
            "4. In Transit\n"
            "5. Out for Delivery\n"
            "6. Delivered\n\n"
            "If the tracking status has not updated for more than 48 hours, contact our support team with your Order ID. "
            "For COD orders, tracking is activated once the courier collects the package. "
            "Guest users can track orders using the Order ID from their confirmation email."
        )
    },
    {
        "id": "doc_005",
        "topic": "Cancellation Policy",
        "text": (
            "You may cancel an order before it is dispatched without any charges.\n\n"
            "How to cancel an order:\n"
            "1. Go to 'My Orders' in your account.\n"
            "2. Select the order you wish to cancel.\n"
            "3. Click 'Cancel Order' and choose a cancellation reason.\n"
            "4. Confirm the cancellation.\n\n"
            "Key cancellation rules:\n"
            "- Orders can be cancelled only before the 'Dispatched' status is assigned.\n"
            "- Once dispatched, cancellation is not possible — you must initiate a return after delivery.\n"
            "- COD orders can be cancelled at no cost at any pre-dispatch stage.\n"
            "- Prepaid orders cancelled before dispatch receive a full refund to the original payment method.\n"
            "- Certain items (e.g., customized products, perishables) are non-cancellable once confirmed.\n\n"
            "If the 'Cancel Order' button is not visible, the order has likely already been dispatched. "
            "In that case, you may refuse delivery or use the return process. "
            "Refunds for cancelled orders are processed within 5–7 business days."
        )
    },
    {
        "id": "doc_006",
        "topic": "Return Policy",
        "text": (
            "Our return policy allows customers to return most products within 7–30 days of delivery, depending on the product category.\n\n"
            "Return eligibility rules:\n"
            "- Item must be unused, unwashed, and in original packaging with all tags intact.\n"
            "- Return must be initiated within the return window shown on the product page.\n"
            "- Items marked 'Non-Returnable' on the product listing are not eligible.\n\n"
            "Non-returnable categories include:\n"
            "- Innerwear, lingerie, and swimwear\n"
            "- Perishable goods and consumables\n"
            "- Customized or personalized products\n"
            "- Digital downloads and software licenses\n\n"
            "How to initiate a return:\n"
            "1. Go to 'My Orders' and select the delivered order.\n"
            "2. Click 'Return' and select the items and reason.\n"
            "3. Choose a pickup date from the available slots.\n\n"
            "Our logistics partner will collect the item from your address. "
            "Once the returned item passes quality inspection at our warehouse, a refund or exchange is processed. "
            "Damaged, used, or incomplete returns may be rejected."
        )
    },
    {
        "id": "doc_007",
        "topic": "Refund Process",
        "text": (
            "Refunds are issued after a return is approved or an order is successfully cancelled before dispatch.\n\n"
            "Refund timelines by payment method:\n"
            "- Credit/Debit Card: 5–7 business days after refund initiation\n"
            "- UPI / Net Banking: 3–5 business days\n"
            "- Digital Wallets: 1–2 business days\n"
            "- Store Credit / Wallet: Instant (if opted)\n"
            "- COD Orders: Refunded to your bank account via NEFT — requires submission of bank details; takes 7–10 business days\n\n"
            "Important refund notes:\n"
            "- Refunds are credited to the original payment source only.\n"
            "- Shipping charges are non-refundable unless the return is due to our error (wrong/damaged item).\n"
            "- You will receive an email notification when your refund is initiated.\n"
            "- Refund status can be tracked under 'My Orders' → 'Refund Status'.\n\n"
            "If the refund does not reflect after the stated timeline, first check with your bank. "
            "If still unresolved, contact our support team with your Order ID and transaction reference."
        )
    },
    {
        "id": "doc_008",
        "topic": "Exchange Policy",
        "text": (
            "An exchange allows you to swap a delivered product for a different size, color, or variant of the same item.\n\n"
            "Exchange eligibility:\n"
            "- Exchange must be requested within 7 days of delivery.\n"
            "- Item must be unused, in original condition with tags and packaging intact.\n"
            "- Exchanges are subject to stock availability of the desired variant.\n\n"
            "How to request an exchange:\n"
            "1. Go to 'My Orders' and select the delivered order.\n"
            "2. Click 'Exchange' and select the item and reason.\n"
            "3. Choose the new size/color/variant.\n"
            "4. Schedule a pickup for the original item.\n\n"
            "Key exchange rules:\n"
            "- Only one exchange is allowed per order item.\n"
            "- If the desired variant is unavailable, you may opt for a refund instead.\n"
            "- Price differences in exchanges: if the new item costs more, you pay the difference; if less, a refund is issued for the difference.\n"
            "- Non-returnable items are also non-exchangeable.\n\n"
            "Exchanged items are dispatched after the original item is picked up and verified."
        )
    },
    {
        "id": "doc_009",
        "topic": "Account Management",
        "text": (
            "Your account is your central hub for managing orders, addresses, payments, and preferences.\n\n"
            "Key account features:\n"
            "- My Orders: View order history, track shipments, initiate returns or exchanges.\n"
            "- Saved Addresses: Add, edit, or delete delivery addresses.\n"
            "- Payment Methods: Save cards or UPI IDs for faster checkout.\n"
            "- Wishlist: Save products you intend to buy later.\n"
            "- Account Settings: Update name, email, phone number, and password.\n"
            "- Notifications: Manage email and SMS alert preferences.\n\n"
            "How to manage your account:\n"
            "- Visit the website or app → Click on your profile icon → Select the relevant section.\n\n"
            "Security tips:\n"
            "- Use a strong, unique password and enable two-factor authentication (2FA) if available.\n"
            "- Never share your OTP or password with anyone, including our support agents.\n\n"
            "To delete your account, contact customer support — account deletion is permanent and removes all order history. "
            "If you forget your password, use the 'Forgot Password' link on the login page to reset it via email or OTP."
        )
    },
    {
        "id": "doc_010",
        "topic": "Discounts & Coupons",
        "text": (
            "We offer various discount types to help customers save on purchases.\n\n"
            "Types of discounts available:\n"
            "- Coupon Codes: Alphanumeric codes entered at checkout for a percentage or flat discount.\n"
            "- Bank Offers: Additional discounts with specific credit/debit cards (e.g., 10% off with HDFC cards).\n"
            "- Loyalty Points: Earned on every purchase; redeemable on future orders (1 point = ₹1).\n"
            "- Referral Discounts: Earn store credit when a referred friend completes their first purchase.\n"
            "- Seasonal Sales: Auto-applied discounts during events like End-of-Season Sale or Diwali Sale.\n\n"
            "How to apply a coupon code:\n"
            "1. Add items to your cart and proceed to checkout.\n"
            "2. Enter the coupon code in the 'Apply Coupon' field.\n"
            "3. Click 'Apply' — the discount reflects in your order total instantly.\n\n"
            "Coupon rules:\n"
            "- Only one coupon can be applied per order.\n"
            "- Coupons are not combinable with each other but may stack with bank offers.\n"
            "- Expired or invalid coupons will display an error message.\n"
            "- Coupons cannot be applied to already-discounted items unless explicitly stated."
        )
    },
    {
        "id": "doc_011",
        "topic": "Customer Support Contact",
        "text": (
            "Our customer support team is available to assist with orders, returns, payments, and any other queries.\n\n"
            "Contact channels:\n"
            "- Live Chat: Available on the website and app, Monday–Saturday, 9 AM–9 PM IST. Average response time: under 2 minutes.\n"
            "- Email: support@ourstore.com — responses within 24 business hours.\n"
            "- Phone Helpline: 1800-XXX-XXXX (toll-free), Monday–Saturday, 9 AM–7 PM IST.\n"
            "- Help Center: Self-service FAQs and guides available 24/7 at help.ourstore.com.\n"
            "- Social Media: DM us on Twitter (@OurStoreSupport) or Instagram for non-urgent queries.\n\n"
            "Before contacting support, keep the following ready:\n"
            "- Your registered email address or phone number\n"
            "- Order ID\n"
            "- Description of the issue with any supporting photos (for damaged/wrong items)\n\n"
            "For faster resolution, use Live Chat for order-related issues. "
            "Escalations can be requested if your issue is not resolved within 48 hours. "
            "We do not offer support via WhatsApp at this time."
        )
    },
    {
        "id": "doc_012",
        "topic": "Common Issues: Failed Payment & Delayed Delivery",
        "text": (
            "This document covers the two most frequently reported issues: payment failures and delivery delays.\n\n"
            "Failed Payment:\n"
            "A failed payment means the transaction did not go through. Common causes:\n"
            "- Incorrect card details or expired card\n"
            "- Insufficient balance or credit limit\n"
            "- Bank-side transaction decline or OTP timeout\n"
            "- Unstable internet connection during payment\n\n"
            "What to do:\n"
            "- Check your bank statement — if money was debited but the order was not placed, it will be auto-refunded within 5–7 business days.\n"
            "- Retry with a different payment method.\n"
            "- Contact your bank if the issue persists.\n\n"
            "Delayed Delivery:\n"
            "A delivery is considered delayed if it has not arrived after the maximum estimated delivery window.\n\n"
            "Common causes:\n"
            "- Incorrect or incomplete delivery address\n"
            "- Local weather disruptions or logistics backlogs\n"
            "- High demand during sale events\n\n"
            "What to do:\n"
            "- Check real-time tracking for the latest status update.\n"
            "- If status is stale for 48+ hours, contact support with your Order ID.\n"
            "- Our team will coordinate with the courier and provide an updated delivery estimate within 24 hours."
        )
    }
]

# embeddings setup
documents_text = [doc["text"] for doc in knowledge_base]
documents_ids = [doc["id"] for doc in knowledge_base]
documents_meta = [{"topic": doc["topic"]} for doc in knowledge_base]

embeddings = embedder.encode(documents_text).tolist()

if len(collection.get()["ids"]) == 0:
    collection.add(
        documents=documents_text,
        ids=documents_ids,
        metadatas=documents_meta,
        embeddings=embeddings
    )

# STATE
class CapstoneState(TypedDict):
    question: str
    retrieved_docs: List[str]
    retrieved_topics: List[str]
    sources: List[str]
    answer: str

# RETRIEVAL
def retrieval_node(state):
    q = state["question"]
    results = collection.query(query_texts=[q], n_results=3)

    state["retrieved_docs"] = results["documents"][0]
    state["retrieved_topics"] = [m["topic"] for m in results["metadatas"][0]]
    state["sources"] = results["ids"][0]
    return state

# ANSWER
def answer_node(state):
    docs = state["retrieved_docs"]
    state["answer"] = "Based on our policy:\n\n" + docs[0]
    return state

# GRAPH
graph = StateGraph(CapstoneState)
graph.add_node("retrieve", retrieval_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")

app_graph = graph.compile(checkpointer=MemorySaver())

# ASK
def ask(question):
    return app_graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": "user1"}}
    )