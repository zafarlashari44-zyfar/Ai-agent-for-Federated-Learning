# MediCare Pro n8n Workflows

## Notification Bridge

Import `workflows/medicare-notification-bridge.json` into the local n8n
instance and publish it.

The production webhook is:

```text
POST http://localhost:5678/webhook/medicare-events
```

The workflow:

1. Validates the `Authorization: Bearer <secret>` header.
2. Responds directly to `automation.health` test events.
3. Forwards supported healthcare events to:
   `POST http://localhost:3000/api/webhooks/n8n`.
4. The Next.js callback validates the event and writes a live Supabase
   notification.

Supported application events:

- `appointment.reminder`
- `prescription.refill`
- `prescription.created`
- `report.ready`
- `patient.registered`
- `billing.alert`

Do not add patient medical details to workflow execution logs. Send only the
minimum identifiers and notification text required for the automation.
