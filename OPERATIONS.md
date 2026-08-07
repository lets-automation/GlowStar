# Running the Server — A Practical Guide

**For:** Glow Star IT
**From:** Lets Automation
**Assumes:** you have never used Linux before. That is fine. You do not need to
learn Linux — you need about ten commands, and they are all in this document.

> **The one-line summary:** the system runs itself. Your job is to check once a
> week that the nightly job is still running, because that is the only thing that
> quietly degrades pricing accuracy if it stops.

---

## 1. Connecting to the server

### 1.1 From a Windows machine

Windows 10 and 11 have this built in — you do not need to install PuTTY.

1. Open **Windows Terminal** (or Command Prompt / PowerShell).
2. Type this, with the username and address we give you:

```
ssh glowstaradmin@<server-address>
```

3. The first time only, it asks *"Are you sure you want to continue connecting?"*
   Type `yes` and press Enter.
4. Enter the password (or it connects automatically if we set up a key).

You are now "on" the server. Everything you type goes to the server, not your PC.

**To leave:** type `exit` and press Enter.

### 1.2 Things that confuse everyone on day one

| What you notice | What is actually happening |
|---|---|
| The password does not appear as you type — not even dots | Normal. Linux hides it completely. Type it and press Enter. |
| There is no mouse, no windows, no icons | Correct. This is a text-only screen. |
| The screen looks like `glowstaradmin@glowstar:~$` | That is the prompt, waiting for a command. |
| You typed something wrong | Press **Ctrl+C** to cancel and get a fresh prompt. |
| Text is scrolling and will not stop | Press **Ctrl+C**. If you are in a log viewer, press **q**. |
| Nothing happened after a command | On Linux, **success is usually silent.** No news is good news. |

**Copy and paste:** select text with the mouse to copy; **right-click** to paste.
`Ctrl+V` does not work in most terminals.

---

## 2. The daily and weekly check

### 2.1 Is the system alive? (10 seconds)

```
curl localhost:8000/health
```

You want to see `"status": "ok"` near the start of the reply.

⚠️ **Important:** this only proves the service is *running*. It does **not**
prove pricing is correct. For that, see 2.3.

### 2.2 Did the nightly job run? (the important one)

**This is the check that matters most.** The nightly job refreshes your price
grid. If it stops, prices slowly drift out of date and nobody notices — that is
exactly what happened on the old Windows laptop.

```
systemctl list-timers glowstar-nightly
```

You will see something like:

```
NEXT                        LEFT       LAST                        PASSED  UNIT
Wed 2026-08-12 02:30:00 IST 9h left    Tue 2026-08-11 02:30:14 IST 14h ago glowstar-nightly.timer
```

| Column | What to check |
|---|---|
| **LAST** / **PASSED** | Should be **less than about 26 hours ago**. If it says two days or more, something is wrong — go to section 4. |
| **NEXT** | Should be tomorrow at 02:30. |

**Do this once a week.** Put it in someone's calendar. It takes ten seconds and
it is the single highest-value thing you can do for pricing accuracy.

### 2.3 Is pricing actually working? (once a week, or after any change)

```
curl -s -X POST localhost:8000/price \
  -H "X-API-Key: YOUR_KEY_HERE" \
  -H 'Content-Type: application/json' \
  -d '{"Shape_full":"ROUND","Weight":1.01,"Color":"G","Clarity":"VS1"}'
```

You should get a price back with a `suggested_discount` in it. If you get an
error instead, the service is running but pricing is broken — go to section 4.

---

## 3. The commands you will actually use

Copy these. You do not need to memorise or understand them.

| What you want | Command |
|---|---|
| Is the pricing service running? | `systemctl status glowstar-api` |
| Restart the pricing service | `sudo systemctl restart glowstar-api` |
| See what the service is doing right now | `journalctl -u glowstar-api -f` (press **q** to exit) |
| See today's errors only | `journalctl -u glowstar-api --since today -p err` |
| Did the nightly job run? | `systemctl list-timers glowstar-nightly` |
| What did the nightly job do last night? | `journalctl -u glowstar-nightly --since yesterday` |
| Run the nightly job right now | `sudo systemctl start glowstar-nightly` |
| How much disk is left? | `df -h /` |
| How much memory is in use? | `free -h` |
| Are the backups being made? | `ls -lh /var/backups/glowstar` |

**Reading `systemctl status`:** look for the word **`active (running)`** in
green. If it says `failed` or `inactive`, see section 4.

---

## 4. When something is wrong

Work down this table. Each row is: what you see → what to run → what it means.

### 4.1 The CRM says it cannot reach the pricing service

```
systemctl status glowstar-api
```

| It says | Meaning | Do this |
|---|---|---|
| `active (running)` | The service is fine — the problem is the network, the domain or the API key | Check section 4.4 |
| `failed` or `inactive` | The service stopped | `sudo systemctl restart glowstar-api`, then check it again |
| `activating` and cycling | It is crashing and restarting in a loop | `journalctl -u glowstar-api -n 50` and send us the output |

### 4.2 The nightly job has not run for two days or more

```
journalctl -u glowstar-nightly --since "3 days ago"
```

| What you see | Meaning | Do this |
|---|---|---|
| Nothing at all | The timer is not firing | `sudo systemctl enable --now glowstar-nightly.timer` |
| Errors about the network or a website | It could not reach a data source | Usually temporary. If it repeats two nights running, tell us |
| `No space left on device` | The disk is full | See 4.3 |

**To catch up immediately**, you can run it by hand at any time:

```
sudo systemctl start glowstar-nightly
```

It takes 5–10 minutes. It is safe to run during the day and safe to run twice.

### 4.3 The disk is filling up

```
df -h /
```

If **Use%** is above 85%, remove backups older than 30 days:

```
sudo find /var/backups/glowstar -name '*.gz' -mtime +30 -delete
```

⚠️ **Never delete anything inside `/opt/glowstar/data/master_grid/`.** See
section 5.

### 4.4 Prices come back but they look wrong

Do **not** restart anything. Restarting hides the evidence. Instead run:

```
curl localhost:8000/health
```

Look at the **model version** and the **data age** in the reply, and send us
that. It tells us immediately whether the model is stale, the grid is stale, or
the pricing itself is at fault.

### 4.5 If you are not sure

**Send us this, and we can almost always diagnose it without logging in:**

```
systemctl status glowstar-api --no-pager
systemctl list-timers glowstar-nightly --no-pager
journalctl -u glowstar-api -n 50 --no-pager
df -h /
```

Copy the output into an email. That is enough for us to work from.

---

## 5. Things that must never be done

| Do not | Why |
|---|---|
| **Delete `/opt/glowstar/data/master_grid/history.json`** | This is the record of what every price cell read on every past day. Your grid API only serves a short recent window, so **a deleted day is gone forever** — and without it, the pricing model cannot be honestly tested. It is the one file in this system that cannot be rebuilt. |
| **Turn the server off, or let it sleep** | The nightly job cannot run. Stale grid data is the single largest cause of pricing error we have measured. |
| **Edit files inside `/opt/glowstar`** | Your changes will be overwritten at the next update, and may break pricing silently. Ask us instead. |
| **Share the API key** | It is the only thing protecting the pricing endpoint. Treat it like a password. If it leaks, tell us and we will issue a new one. |
| **Run commands as `root` that you found on the internet** | Standard advice, but worth repeating. Everything you legitimately need is in this document. |

---

## 6. Routine maintenance

| How often | What | Command |
|---|---|---|
| **Weekly** | Confirm the nightly job ran | `systemctl list-timers glowstar-nightly` |
| **Weekly** | Confirm backups exist | `ls -lh /var/backups/glowstar` |
| Monthly | Check disk space | `df -h /` |
| Monthly | Apply security updates | `sudo apt update && sudo apt upgrade -y` |
| After any reboot | Confirm both came back | `systemctl status glowstar-api` and `systemctl list-timers glowstar-nightly` |

**Security updates and reboots:** Ubuntu occasionally needs a restart after
updates. That is safe. Both the pricing service and the nightly timer start
automatically when the server boots — you do not need to start them by hand. If
the machine was off at 02:30, the nightly job runs as soon as it comes back.

---

## 7. A short glossary

| Term | Plain meaning |
|---|---|
| **SSH** | The way you connect to the server — like Remote Desktop, but text only |
| **systemd / systemctl** | Linux's service manager. It starts our programs, restarts them if they crash, and keeps the logs |
| **service** | A program that runs in the background — ours is `glowstar-api` |
| **timer** | Linux's scheduled task, equivalent to Windows Task Scheduler — ours is `glowstar-nightly` |
| **journalctl** | The log viewer |
| **sudo** | "Do this as administrator". Needed for restarting services and installing updates |
| **`/opt/glowstar`** | The folder where everything of ours lives |
| **nginx** | The web server that handles HTTPS and passes requests to our pricing service |
| **PostgreSQL** | The database that stores every price we quoted and every decision your desk made |

---

## 8. Who to contact

**Lets Automation** — for anything in section 4 that is not resolved by a
restart, and for anything at all in section 4.4 (prices look wrong).

Please include the output from **section 4.5** in the first message. It saves a
round trip and usually lets us answer straight away.
