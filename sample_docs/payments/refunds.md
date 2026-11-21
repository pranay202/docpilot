# Refunds

You can refund a succeeded payment fully or partially.

## Create a refund

Call `client.refunds.create(payment_intent="pi_123", amount=2000)` to refund 20.00 of the original payment.
Leave out `amount` to refund the full payment.
A payment can be refunded multiple times until the total refunded amount equals the original amount.

## Refund timelines

Card refunds reach the customer in 5 to 7 business days. UPI refunds usually complete within 48 hours.
Refunds cannot be cancelled once they are created.
