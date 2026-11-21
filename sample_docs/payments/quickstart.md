# Payments Quickstart

The Payments API lets you accept card and UPI payments with a single integration.

## Install the SDK

1. Install the Python SDK with `pip install acmepay`.
2. Create an API key in the dashboard under Settings > API keys.
3. Export the key as the `ACMEPAY_API_KEY` environment variable.
4. Create a client with `acmepay.Client()`; it reads the key from the environment.

## Create a payment intent

A payment intent tracks one payment from creation to settlement.
Call `client.payment_intents.create(amount=5000, currency="INR")`. Amounts are in the smallest currency unit, so 5000 means 50.00 INR.
The response contains a `client_secret` that the frontend uses to confirm the payment.

## Test mode

Keys that start with `sk_test_` run in test mode and never move real money.
Use card number 4242 4242 4242 4242 with any future expiry date to simulate a successful payment.
