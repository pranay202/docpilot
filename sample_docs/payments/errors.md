# Errors and rate limits

## Error format

Errors return a JSON body with `code`, `message` and `request_id`. Include the `request_id` when you contact support.

## Rate limits

The API allows 100 requests per second per API key in live mode and 25 requests per second in test mode.
Requests over the limit receive status 429 with a `Retry-After` header that says how many seconds to wait.

## Idempotency

Send an `Idempotency-Key` header on POST requests so that a retried request does not create a duplicate payment. Keys expire after 24 hours.
