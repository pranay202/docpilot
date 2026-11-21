# Notifications Overview

The Notifications API sends SMS, email and push messages from one endpoint.

## Set up the notifications service

1. Enable the Notifications product in the dashboard.
2. Verify a sender domain for email by adding the DNS TXT record shown in the dashboard.
3. Register an SMS sender ID; approval takes up to 2 business days.
4. Create an API key with the `notifications:send` scope.

## Send a message

Call `POST /v1/messages` with `channel` set to `sms`, `email` or `push`, a `to` address and a `template_id`.
Template variables are passed in the `data` object.
