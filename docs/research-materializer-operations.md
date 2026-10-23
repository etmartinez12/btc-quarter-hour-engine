# Production research materializer operations

The `btc-qh-research-materializer.timer` runs the one-shot
`btc-qh-research-materializer.service` at **HH:10 UTC every hour**. Each run
processes at most 100 new sealed segments and checkpoints every 25 segments.
The production collector remains the higher-priority workload; this service
uses lower CPU and I/O weights and a lower scheduling priority.

The service reads the production raw data lake at
`/var/lib/btc-qh/data_lake_prod` and writes derived artifacts and its
continuation checkpoint under
`/var/lib/btc-qh/research_materializer_prod`. The raw input is mounted
read-only to this service. The checkpoint is guarded by
`ConditionPathExists=`: if
`/var/lib/btc-qh/research_materializer_prod/checkpoint.json` is absent, systemd
will skip execution. **Automatic seeding is not implemented.** An operator
must first seed the operational directory from the verified PR #24/PR #25
continuation state; an empty directory must never cause automatic replay from
raw segment zero.

## Deployment and validation

Perform these steps manually, in order:

A. Merge PR #26.
B. Update the production checkout to the merged `main` SHA without restarting
   the collector.
C. Create `/var/lib/btc-qh/research_materializer_prod`.
D. Manually seed it from the verified 2,331-segment PR #24/PR #25 continuation
   state. Do not replace an existing operational checkpoint.
E. Before installing or running the service, verify the seeded checkpoint:
   version `1`, `2,331` processed segments, `21,639,009` frames, and session
   `acfada3f-da3c-4aa9-8540-56760cd8eba3`.
F. Copy the unit files into systemd's unit directory:

   ```bash
   sudo install -m 0644 deploy/systemd/btc-qh-research-materializer.service /etc/systemd/system/
   sudo install -m 0644 deploy/systemd/btc-qh-research-materializer.timer /etc/systemd/system/
   ```

G. Reload systemd's unit definitions:

   ```bash
   sudo systemctl daemon-reload
   ```

H. Validate the service manually before enabling the timer:

   ```bash
   sudo systemctl start btc-qh-research-materializer.service
   sudo systemctl status btc-qh-research-materializer.service
   ```

I. Verify `btc-qh-collector.service` remained active throughout; this procedure
   does not stop, restart, or modify the production collector.
J. Inspect the materializer journal and resulting checkpoint/artifacts:

   ```bash
   sudo journalctl -u btc-qh-research-materializer.service
   ```

K. Only after successful manual validation, enable and start the timer:

   ```bash
   sudo systemctl enable --now btc-qh-research-materializer.timer
   ```

Inspect service and timer state with:

```bash
sudo systemctl status btc-qh-research-materializer.service
sudo systemctl status btc-qh-research-materializer.timer
sudo systemctl list-timers btc-qh-research-materializer.timer
```

To manually trigger a run, use
`sudo systemctl start btc-qh-research-materializer.service`. To disable future
scheduled runs, use
`sudo systemctl disable --now btc-qh-research-materializer.timer`; this does
not stop or restart the collector.

## Locking and recovery

The service invokes the materializer through `/usr/bin/flock` using
`/var/lib/btc-qh/research_materializer.lock` with nonblocking acquisition. If
another materializer owns the lock, `flock` exits with status 75 and systemd
treats that result as a successful no-op. The lock is released when the
invocation exits. Ensure the service user can create/use the lock and write
the operational output directory.

Actual checkpoint, raw-prefix, replay, or artifact errors remain nonzero
service failures and are recorded in the journal. There is no automatic
restart loop. Inspect the service status and journal, resolve the underlying
problem without deleting or overwriting the operational checkpoint, then
trigger a manual run. The timer retries on its next hourly schedule. With
`Persistent=true`, a timer event missed while the VM was down is run after
systemd starts again.
