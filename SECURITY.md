# Security

## Reporting a vulnerability

Please report security problems privately through GitHub's **"Report a vulnerability"** button on this repository's Security tab (a private security advisory), not in a public issue. Include what you did, what happened and what you expected, and the commit you ran. You can expect an answer within a couple of weeks; this is a one-person research project.

## What is in scope

DiffusionGearvis runs on your own machine. The parts where a security problem matters most:

- **Network exposure.** Every service is meant to bind to 127.0.0.1 only. Anything that listens beyond loopback, or reaches the network outside the documented paths (the one-time downloads, web search, and the agent's browser in web mode through its egress proxy), is a vulnerability.
- **The agent's browser** (`--tools live`). It is meant to run with a fresh profile, without your logins; to refuse loopback, private and link-local addresses, ports other than 443 and plain `http`; never to type into password or card fields, or text that looks like a card number; never to download or upload files; and to ask before buying, sending, posting or deleting. A way around any of these is a vulnerability.
- **Secrets.** The optional Brave Search API key (`~/.config/diffusion_gearvis/brave_api_key`) must only ever go in its request header, never into a log, the console or the model. The text you type into "Your instructions to Gearvis" must only appear in the model requests it is part of.
- **The HUD page.** Text from the model, from web pages or from search results must never run as code in the page.

## Known limits (not vulnerabilities by themselves)

- **Prompt injection.** Web pages and search results are untrusted text, and the model sometimes follows instructions planted in them (on our test page, 4 times in 20; see the README's "Real websites"). The safeguards above are enforced in code so that injected instructions cannot get past them; a report that shows a safeguard failing is welcome, a report that the model can be talked into something within them is a known limit.
- **No echo cancellation.** With speakers instead of a headset, Gearvis can hear itself.
- **Session logs** (`logs/sessions/`) are plain files on your disk and hold what you said and typed. Treat them as private.

## Supported versions

Only the latest release on `main` gets security fixes.
