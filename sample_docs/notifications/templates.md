# Templates

Templates hold the message content so your code only sends variables.

## Create a template

Create templates in the dashboard or with `POST /v1/templates`. Use double curly braces for variables, for example `{{first_name}}`.
SMS templates are limited to 640 characters. Email templates support HTML and a plain-text fallback.

## Localization

Add one version of a template per locale. The API picks the version that matches the recipient's `locale` field and falls back to English.
