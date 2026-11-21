# Webhooks

Webhooks notify your server when a payment changes state, for example `payment.succeeded` or `payment.failed`.

## Register an endpoint

1. Open Settings > Webhooks in the dashboard.
2. Add your HTTPS endpoint URL.
3. Select the events you want to receive.
4. Copy the signing secret shown after saving.

## Verify signatures

Every webhook request carries an `X-AcmePay-Signature` header.
Compute an HMAC-SHA256 of the raw request body using your signing secret and compare it with the header using a constant-time comparison.
Reject the request with status 400 if the signatures do not match.

## Retries

If your endpoint does not return a 2xx status within 10 seconds, the webhook is retried with exponential backoff for up to 3 days.
Make your handler idempotent by storing the event `id` and ignoring duplicates.
