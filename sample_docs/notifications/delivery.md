# Delivery and status

## Delivery statuses

Each message moves through `queued`, `sent`, `delivered` or `failed`.
Fetch the status with `GET /v1/messages/{id}`.

## Quiet hours

Set `quiet_hours` on a recipient to hold non-urgent messages between 22:00 and 08:00 in the recipient's timezone.
Messages marked `priority: high`, such as one-time passwords, ignore quiet hours.

## Opt-outs

Recipients who reply STOP to an SMS are opted out automatically. Sending to an opted-out number returns error code `recipient_opted_out`.
