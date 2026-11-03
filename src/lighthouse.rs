// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use core::net::SocketAddr;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use anyhow::anyhow;
use askama::Template;
use axum::Router;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::Html;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::routing::post;
use gethostname::gethostname;
use log::error;
use log::info;
use log::warn;
use structopt::StructOpt;
use tokio::sync::Mutex;
use tokio::sync::broadcast;
use tokio::task::JoinSet;
use tokio::time::interval;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::service::Routes;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

use crate::manager::manager_client_new;
use crate::manager::manager_control_client_new;
use crate::torchftpb::CommandRequest;
use crate::torchftpb::CommandResponse;
use crate::torchftpb::FaultEvent;
use crate::torchftpb::KillRequest;
use crate::torchftpb::lighthouse_control_service_server::LighthouseControlServiceServer;
use crate::torchftpb::LighthouseHeartbeatRequest;
use crate::torchftpb::LighthouseHeartbeatResponse;
use crate::torchftpb::LighthouseQuorumRequest;
use crate::torchftpb::LighthouseQuorumResponse;
use crate::torchftpb::ManagerCommandRequest;
use crate::torchftpb::MemberInstructionRequest;
use crate::torchftpb::MemberInstructionResponse;
use crate::torchftpb::ObservationRequest;
use crate::torchftpb::ObservationResponse;
use crate::torchftpb::Quorum;
use crate::torchftpb::QuorumMember;
use crate::torchftpb::ReplicaObservation;
use crate::torchftpb::ReportEventRequest;
use crate::torchftpb::ReportEventResponse;
use crate::torchftpb::lighthouse_control_service_server::LighthouseControlService;
use crate::torchftpb::lighthouse_service_server::LighthouseService;
use crate::torchftpb::lighthouse_service_server::LighthouseServiceServer;
use crate::torchftpb::PushEventRequest;

/// Number of events buffered for clusterd consumption. Events dropped beyond
/// this bound are lost to a clusterd that stays down longer than the buffer
/// can cover; acked events are shed from the head eagerly so the capacity is
/// spent on undelivered facts only.
const EVENT_BUFFER_CAPACITY: usize = 4096;

/// Sender-side resend period for un-acked fault events (v5.13 push-ONLY
/// transport). Deliberately generous: PushEvent folds the event
/// synchronously -- command dispatch (SendCommand to managers) happens
/// inside the fold under the engine lock -- so a resend arriving while a
/// clusterd adjudication is still running would only stack duplicate
/// pushes behind it. 15s leaves ample room for any single adjudication to
/// complete; the cost of the slack is that a push-failed event reaches
/// clusterd at most one period late (fresh events still ride the <1ms
/// immediate push).
const EVENT_RESEND_INTERVAL_SECS: u64 = 15;

/// Maximum size of the raw exception string carried in a FaultEvent.
/// Must not be smaller than the wrapper's _RAW_DETAIL_CAP (16KB,
/// tail-preserved -- the root cause of an exception chain sits at the
/// end), otherwise the informative tail the wrapper kept gets chopped
/// off here.
const EVENT_DETAIL_MAX_CHARS: usize = 16384;

#[derive(Clone)]
struct QuorumMemberDetails {
    joined: Instant,
    member: QuorumMember,
}

/// replica identity: full replica_id is "{group_name}:{uuid}" (see
/// torchft/manager.py). The group name is the stable identity clusterd
/// addresses commands at; the uuid disambiguates incarnations after rebuild.
fn group_of(replica_id: &str) -> String {
    match replica_id.split_once(':') {
        Some((group, _)) => group.to_string(),
        None => replica_id.to_string(),
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Tail-preserve detail down to max bytes: keep the end of the string
/// with a head-dropped marker (same policy as the wrapper's
/// _raw_detail). Char-boundary safe.
fn truncate_detail_tail(detail: &mut String, max: usize) {
    if detail.len() <= max {
        return;
    }
    let dropped = detail.len() - max;
    let mut start = detail.len() - (max - 64);
    while !detail.is_char_boundary(start) {
        start -= 1;
    }
    let tail = detail[start..].to_string();
    *detail = format!("...({}B head dropped)...\n{}", dropped, tail);
}

struct State {
    channel: broadcast::Sender<Quorum>,
    participants: HashMap<String, QuorumMemberDetails>,
    prev_quorum: Option<Quorum>,
    quorum_id: i64,

    // heartbeat information
    // replica_id -> last heartbeat
    heartbeats: HashMap<String, Instant>,

    // ==== MC-1/MC-9 control plane state ====
    // event buffer for clusterd (contract 3 upstream relay)
    events: VecDeque<FaultEvent>,
    next_event_seq: u64,
    // seqs acked by clusterd's sink (v5.13 push-only: an acked event is
    // never resent; un-acked ones are redelivered by the resend sweep)
    event_acked: HashSet<u64>,
    // group names excluded by clusterd member instructions
    excluded: HashSet<String>,
    // replica_ids seen before -- a never-seen id is a new incarnation
    // (rejoin after rebuild) and emits a replica_joined event
    known_replicas: HashSet<String>,
    // replica groups considered alive at the previous quorum tick
    last_alive: HashSet<String>,
    // member instruction ids already applied (idempotency)
    applied_instructions: HashSet<String>,
}

pub struct Lighthouse {
    state: Mutex<State>,
    opt: LighthouseOpt,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
    change_logger: ChangeLogger,
}

struct ChangeLogger {
    last_reason: std::sync::Mutex<Option<String>>,
}
impl ChangeLogger {
    fn new() -> Self {
        ChangeLogger {
            last_reason: std::sync::Mutex::new(None),
        }
    }
    fn log_if_changed(&self, reason: &str) {
        let mut last_reason = self.last_reason.lock().unwrap();
        if last_reason.as_deref() != Some(reason) {
            info!("Quorum status: {}", reason);
            *last_reason = Some(reason.to_string());
        }
    }
}

#[derive(StructOpt, Debug)]
#[structopt()]
pub struct LighthouseOpt {
    // bind is the address to bind the server to.
    #[structopt(
        long = "bind",
        default_value = "[::]:29510",
        help = "Address to bind the server to"
    )]
    pub bind: String,

    #[structopt(
        long = "join_timeout_ms",
        default_value = "60000",
        help = "How long to wait for heartbeating stragglers to join before issuing quorum"
    )]
    pub join_timeout_ms: u64,

    #[structopt(
        long = "min_replicas",
        help = "Minimum number of replicas to consider a quorum"
    )]
    pub min_replicas: u64,

    #[structopt(
        long = "quorum_tick_ms",
        default_value = "100",
        help = "How frequently to check for quorum when waiting for stragglers."
    )]
    pub quorum_tick_ms: u64,

    #[structopt(
        long = "heartbeat_timeout_ms",
        default_value = "5000",
        help = "How long to wait for a heartbeat before considering a replica dead."
    )]
    pub heartbeat_timeout_ms: u64,

    // Contract 3 fast path: when set (e.g. http://<node>:29600), every
    // buffered fault event is pushed fire-and-forget to clusterd's
    // ClusterdEventSink. Empty (default) disables pushing -- clusterd
    // then consumes events via the GetEvents pull only.
    #[structopt(
        long = "clusterd-addr",
        default_value = "",
        help = "clusterd ClusterdEventSink address for event push; empty disables push (pull only)"
    )]
    pub clusterd_addr: String,
}

fn quorum_changed(a: &Vec<QuorumMember>, b: &Vec<QuorumMember>) -> bool {
    let a_ids: Vec<&String> = a.iter().map(|p| &p.replica_id).collect();
    let b_ids: Vec<&String> = b.iter().map(|p| &p.replica_id).collect();

    return a_ids != b_ids;
}

// Checks whether the quorum is valid, the new quorum and an explanation for the state.
fn quorum_compute(
    now: Instant,
    state: &State,
    opt: &LighthouseOpt,
) -> (Option<Vec<QuorumMember>>, String) {
    let heartbeats = &state.heartbeats;
    let healthy_replicas: HashSet<&String> = heartbeats
        .iter()
        .filter_map(|(replica_id, last_heartbeat)| {
            if now.duration_since(*last_heartbeat) < Duration::from_millis(opt.heartbeat_timeout_ms)
            {
                return Some(replica_id);
            }
            None
        })
        .collect();

    // MC-9 design rule: a clusterd member instruction can only ever tighten
    // the quorum. Excluded replicas are dropped from the candidate set no
    // matter how healthy they look; restore goes through SetMemberList and
    // then still has to pass the normal heartbeat/majority checks below.
    let healthy_participants: HashMap<&String, &QuorumMemberDetails> = state
        .participants
        .iter()
        .filter(|(replica_id, _details)| healthy_replicas.contains(replica_id))
        .filter(|(replica_id, _details)| !state.excluded.contains(&group_of(replica_id)))
        .collect();

    let mut candidate_participants: Vec<QuorumMember> = healthy_participants
        .values()
        .map(|details| details.member.clone())
        .collect();

    // Sort by replica ID to get a consistent ordering across runs.
    candidate_participants.sort_by_key(|p| p.replica_id.clone());

    let shrink_only = healthy_participants
        .iter()
        .any(|(_, details)| details.member.shrink_only);

    let metadata = format!(
        "[{}/{} participants healthy][{} heartbeating][shrink_only={}]",
        healthy_participants.len(),
        state.participants.len(),
        healthy_replicas.len(),
        shrink_only,
    );

    // Check if we can use the previous quorum.
    // TODO: do we still need this given we have heartbeats?
    if state.prev_quorum.is_some() {
        let prev_quorum = state.prev_quorum.as_ref().unwrap();

        let prev_replica_ids: HashSet<&String> = prev_quorum
            .participants
            .iter()
            .map(|p| &p.replica_id)
            .collect();

        if shrink_only {
            candidate_participants = candidate_participants
                .into_iter()
                .filter(|p| prev_replica_ids.contains(&p.replica_id))
                .collect();
        }

        // Fast quorum is when all previous participants are still in the quorum
        // and we have enough participants to form a quorum.
        let is_fast_quorum = prev_quorum
            .participants
            .iter()
            .all(|prev_member| healthy_participants.contains_key(&prev_member.replica_id));

        if is_fast_quorum {
            return (
                Some(candidate_participants),
                format!("Fast quorum found! {}", metadata),
            );
        }
    }

    // Minimum quorum size check.
    if healthy_participants.len() < opt.min_replicas as usize {
        return (
            None,
            format!(
                "New quorum not ready, only have {} participants, need min_replicas {} {}",
                healthy_participants.len(),
                opt.min_replicas,
                metadata
            ),
        );
    }

    // Avoid split brain by requiring at least half of the known alive workers.
    if healthy_participants.len() <= healthy_replicas.len() / 2 {
        return (
            None,
            format!(
                "New quorum not ready, only have {} participants, need at least half of {} healthy workers {}",
                healthy_participants.len(),
                healthy_replicas.len(),
                metadata
            ),
        );
    }

    let all_healthy_joined = healthy_participants.len() == healthy_replicas.len();

    // Quorum is valid at this point but lets wait for stragglers.
    let first_joined = healthy_participants
        .values()
        .map(|details| details.joined)
        .min()
        .unwrap_or(now);
    if !all_healthy_joined
        && now.duration_since(first_joined) < Duration::from_millis(opt.join_timeout_ms)
    {
        return (
            None,
            format!(
                "Valid quorum with {} participants, waiting for {} healthy but not participating stragglers due to join timeout {}",
                healthy_participants.len(),
                healthy_replicas.len() - healthy_participants.len(),
                metadata
            ),
        );
    }

    (
        Some(candidate_participants),
        format!("Valid quorum found {}", metadata),
    )
}

impl Lighthouse {
    pub async fn new(opt: LighthouseOpt) -> Result<Arc<Self>> {
        let listener = tokio::net::TcpListener::bind(&opt.bind).await?;

        let (tx, _) = broadcast::channel(16);

        Ok(Arc::new(Self {
            state: Mutex::new(State {
                participants: HashMap::new(),
                channel: tx,
                prev_quorum: None,
                quorum_id: 0,
                heartbeats: HashMap::new(),
                events: VecDeque::new(),
                next_event_seq: 1,
                event_acked: HashSet::new(),
                excluded: HashSet::new(),
                known_replicas: HashSet::new(),
                last_alive: HashSet::new(),
                applied_instructions: HashSet::new(),
            }),
            opt,
            local_addr: listener.local_addr()?,
            listener: Mutex::new(Some(listener)),
            change_logger: ChangeLogger::new(),
        }))
    }

    fn _quorum_tick(self: Arc<Self>, state: &mut State) -> Result<()> {
        // Detect heartbeat-loss transitions and relay them upstream to
        // clusterd (contract 3): a replica that was alive at the previous
        // tick and is not anymore is a "heartbeat_loss" event. The exclusion
        // decision stays with clusterd -- lighthouse only reports the fact.
        let now = Instant::now();
        let alive: HashSet<String> = state
            .heartbeats
            .iter()
            .filter(|(_, last)| {
                now.duration_since(**last) < Duration::from_millis(self.opt.heartbeat_timeout_ms)
            })
            .map(|(replica_id, _)| replica_id.clone())
            .collect();
        let lost: Vec<String> = state
            .last_alive
            .iter()
            .filter(|replica_id| !alive.contains(*replica_id))
            .cloned()
            .collect();
        for replica_id in lost {
            info!(
                "heartbeat lost for replica {}, relaying to clusterd",
                replica_id
            );
            let event = Self::push_event(
                state,
                FaultEvent {
                    seq: 0,
                    group_rank: 0,
                    source: "heartbeat_loss".to_string(),
                    replica_id: replica_id.clone(),
                    group_name: group_of(&replica_id),
                    error_class: "HEARTBEAT_LOSS".to_string(),
                    error_code: String::new(),
                    window: "none".to_string(),
                    step: -1,
                    retry_count: 0,
                    detail: format!(
                        "replica {} heartbeat absent for more than {}ms",
                        replica_id, self.opt.heartbeat_timeout_ms
                    ),
                    timestamp_ms: now_ms(),
                },
            );
            Self::spawn_event_push(self.opt.clusterd_addr.clone(), event);
        }
        state.last_alive = alive;

        let (quorum_met, reason) = quorum_compute(Instant::now(), state, &self.opt);
        self.change_logger.log_if_changed(&reason);

        if quorum_met.is_some() {
            let participants = quorum_met.unwrap();

            let commit_failure_replica_ids: Vec<String> = participants
                .iter()
                .filter(|p| p.commit_failures > 0)
                .map(|p| p.replica_id.clone())
                .collect();

            // only increment quorum ID if something about the quorum
            // changed (members/addresses/etc)
            if state.prev_quorum.is_none()
                || quorum_changed(
                    &participants,
                    &state.prev_quorum.as_ref().unwrap().participants,
                )
            {
                state.quorum_id += 1;
                info!(
                    "Detected quorum change, bumping quorum_id to {}",
                    state.quorum_id
                );
            } else if commit_failure_replica_ids.len() > 0 {
                state.quorum_id += 1;
                info!(
                    "Detected commit failures in [{}], bumping quorum_id to {}",
                    commit_failure_replica_ids.join(", "),
                    state.quorum_id
                );
            }

            let quorum = Quorum {
                quorum_id: state.quorum_id,
                participants,
                created: Some(SystemTime::now().into()),
            };

            info!("Quorum! {:?}", quorum);

            // Membership changes are relayed upstream (contract 3
            // quorum observation): clusterd reconciles its per-replica state
            // machine against the issued participant set.
            let groups: Vec<String> = quorum
                .participants
                .iter()
                .map(|p| group_of(&p.replica_id))
                .collect();
            if let Some(prev) = &state.prev_quorum {
                let prev_groups: HashSet<String> =
                    prev.participants.iter().map(|p| group_of(&p.replica_id)).collect();
                let current: HashSet<String> = groups.iter().cloned().collect();
                if prev_groups != current {
                    let event = Self::push_event(
                        state,
                        FaultEvent {
                            seq: 0,
                            group_rank: 0,
                            source: "quorum_change".to_string(),
                            replica_id: String::new(),
                            group_name: String::new(),
                            error_class: "QUORUM_CHANGE".to_string(),
                            error_code: String::new(),
                            window: "none".to_string(),
                            step: -1,
                            retry_count: 0,
                            detail: format!(
                                "quorum {:?} -> {:?} (quorum_id={})",
                                prev_groups, current, quorum.quorum_id
                            ),
                            timestamp_ms: now_ms(),
                        },
                    );
                    Self::spawn_event_push(self.opt.clusterd_addr.clone(), event);
                }
            }

            state.prev_quorum = Some(quorum.clone());
            state.participants.clear();
            match state.channel.send(quorum) {
                Ok(_) => (),
                Err(e) => error!("failed to send quorum {}", e),
            }
        }
        Ok(())
    }

    async fn _run_quorum(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.opt.quorum_tick_ms));
        loop {
            interval.tick().await; // Wait for the next tick
            let mut state = self.state.lock().await;
            self.clone()._quorum_tick(&mut state)?;
        }
    }

    pub fn address(&self) -> String {
        format!(
            "http://{}:{}",
            gethostname().into_string().unwrap(),
            self.local_addr.port()
        )
    }

    /// Buffer an event for clusterd consumption (contract 3). seq is
    /// assigned here; the buffer is bounded, oldest events drop first.
    /// Returns the seq-assigned event so callers can immediately push it
    /// to clusterd's sink (the resend sweep covers any push failure).
    /// Acked events are useless state -- shed them from the head first so
    /// the capacity serves undelivered facts.
    fn push_event(state: &mut State, mut event: FaultEvent) -> FaultEvent {
        event.seq = state.next_event_seq;
        state.next_event_seq += 1;
        while state
            .events
            .front()
            .map_or(false, |e| state.event_acked.contains(&e.seq))
        {
            state.events.pop_front();
        }
        if state.events.len() >= EVENT_BUFFER_CAPACITY {
            state.events.pop_front();
        }
        state.events.push_back(event.clone());
        event
    }

    /// Deliver one event to clusterd's event sink and wait for the
    /// verdict. Ok means the sink processed (or dedup-collapsed) the
    /// event -- the transport-level ack for the push-ONLY design; Err
    /// (connect/timeout/Internal) leaves the event un-acked, and the
    /// resend sweep redelivers it after EVENT_RESEND_INTERVAL_SECS.
    async fn push_to_clusterd(clusterd_addr: String, event: FaultEvent) -> Result<()> {
        let mut client = crate::manager::clusterd_event_sink_client_new(
            clusterd_addr,
            Duration::from_secs(2),
        )
        .await?;
        let mut req = Request::new(PushEventRequest { event: Some(event) });
        req.set_timeout(Duration::from_secs(5));
        client
            .push_event(req)
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    }

    /// Fire-and-forget push of a buffered event to clusterd's event sink
    /// (contract 3 fast path -- sub-ms delivery). Ack bookkeeping is NOT
    /// done here: the event stays un-acked, and the resend sweep either
    /// confirms delivery via its own Ok or redelivers; a duplicate
    /// redelivery collapses on mc1d's per-seq processed dedup. A push
    /// failure never fails the caller (ReportEvent reply / quorum tick).
    fn spawn_event_push(clusterd_addr: String, event: FaultEvent) {
        if clusterd_addr.is_empty() {
            return;
        }
        let seq = event.seq;
        tokio::spawn(async move {
            if let Err(e) = Self::push_to_clusterd(clusterd_addr, event).await {
                warn!(
                    "event push to clusterd failed (seq {}, resend sweep will redeliver): {}",
                    seq, e
                );
            }
        });
    }

    /// Sender-side resend sweep for the push-ONLY event transport: every
    /// EVENT_RESEND_INTERVAL_SECS, re-push every buffered-but-un-acked
    /// event; an Ok response acks the seq (never resent again). This is
    /// the whole failure-recovery story -- mc1d restarts and failed
    /// pushes are healed by redelivery, no pull path exists.
    async fn _run_event_resend(self: Arc<Self>) -> Result<()> {
        let mut ticker = tokio::time::interval(Duration::from_secs(EVENT_RESEND_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let clusterd_addr = self.opt.clusterd_addr.clone();
            if clusterd_addr.is_empty() {
                continue;
            }
            let pending: Vec<FaultEvent> = {
                let state = self.state.lock().await;
                state
                    .events
                    .iter()
                    .filter(|e| !state.event_acked.contains(&e.seq))
                    .cloned()
                    .collect()
            };
            if pending.is_empty() {
                continue;
            }
            info!(
                "event resend sweep: {} un-acked event(s) queued for redelivery",
                pending.len()
            );
            for event in pending {
                let clusterd_addr = clusterd_addr.clone();
                let this = self.clone();
                tokio::spawn(async move {
                    let seq = event.seq;
                    match Self::push_to_clusterd(clusterd_addr, event).await {
                        Ok(()) => {
                            // transport ack: never resend this seq again
                            this.state.lock().await.event_acked.insert(seq);
                        }
                        Err(e) => {
                            warn!("event resend to clusterd failed (seq {}): {}", seq, e);
                        }
                    }
                });
            }
        }
    }

    /// Look up the ManagerServer address of a replica group (most recent
    /// incarnation seen), for command forwarding (contract 8 downlink).
    fn manager_address_for(state: &State, group_name: &str) -> Option<String> {
        let mut best: Option<(Instant, String)> = None;
        for (replica_id, details) in &state.participants {
            if group_of(replica_id) == group_name {
                match &best {
                    Some((joined, _)) if *joined > details.joined => {}
                    _ => best = Some((details.joined, details.member.address.clone())),
                }
            }
        }
        if best.is_none() {
            // fall back to the last issued quorum
            if let Some(prev) = &state.prev_quorum {
                for member in &prev.participants {
                    if group_of(&member.replica_id) == group_name {
                        return Some(member.address.clone());
                    }
                }
            }
        }
        best.map(|(_, addr)| addr)
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info!("Lighthouse listening on: {}", self.address());

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        // Setup HTTP endpoints
        let app = Router::new()
            .route(
                "/",
                get(|| async { Html(IndexTemplate {}.render().unwrap()) }),
            )
            .route(
                "/status",
                get({
                    let self_clone = self.clone();
                    move || async { self_clone.get_status().await }
                }),
            )
            .route(
                "/replica/:replica_id/kill",
                post({
                    let self_clone = self.clone();
                    move |path| async { self_clone.kill(path).await }
                }),
            );

        // register the GRPC services
        let routes = Routes::from(app)
            .add_service(LighthouseServiceServer::new(self.clone()))
            .add_service(LighthouseControlServiceServer::new(self));

        Server::builder()
            // allow non-GRPC connections
            .accept_http1(true)
            .add_routes(routes)
            .serve_with_incoming(incoming)
            .await
            .map_err(|e| e.into())
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_quorum());

        set.spawn(self.clone()._run_grpc());

        set.spawn(self.clone()._run_event_resend());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }

    async fn get_status(self: Arc<Self>) -> Html<String> {
        let template = {
            let state = self.state.lock().await;

            let (_, quorum_status) = quorum_compute(Instant::now(), &state, &self.opt);

            let max_step = if let Some(quorum) = &state.prev_quorum {
                quorum
                    .participants
                    .iter()
                    .map(|p| p.step)
                    .max()
                    .unwrap_or(-1)
            } else {
                -1
            };

            let num_participants = if let Some(quorum) = &state.prev_quorum {
                quorum.participants.len() as i64
            } else {
                -1
            };

            StatusTemplate {
                quorum_id: state.quorum_id,
                num_participants,
                prev_quorum: state.prev_quorum.clone(),
                quorum_status,
                max_step,

                heartbeats: state.heartbeats.clone(),
                old_age_threshold: Instant::now()
                    .checked_sub(Duration::from_millis(self.opt.heartbeat_timeout_ms))
                    .unwrap_or(Instant::now()),
            }
        };
        Html(template.render().unwrap())
    }

    async fn kill(self: Arc<Self>, Path(replica_id): Path<String>) -> Result<(), AppError> {
        let addr = 'addr: {
            let state = self.state.lock().await;

            if state.prev_quorum.is_none() {
                return Err(AppError(anyhow!("failed to find replica")));
            }

            for member in state.prev_quorum.clone().unwrap().participants {
                if member.replica_id == replica_id {
                    break 'addr member.address;
                }
            }

            return Err(AppError(anyhow!("failed to find replica")));
        };

        let mut client = manager_client_new(addr, Duration::from_secs(10)).await?;

        let request = tonic::Request::new(KillRequest {
            msg: "killed from dashboard".to_string(),
        });
        let _resp = client.kill(request).await?;

        Ok(())
    }
}

#[tonic::async_trait]
impl LighthouseService for Arc<Lighthouse> {
    async fn quorum(
        &self,
        request: Request<LighthouseQuorumRequest>,
    ) -> Result<Response<LighthouseQuorumResponse>, Status> {
        let req = request.into_inner();
        let requester = req
            .requester
            .ok_or_else(|| return Status::invalid_argument("missing requester"))?;

        info!(
            "Received quorum request for replica {}",
            &requester.replica_id
        );

        let mut rx = {
            let mut state = self.state.lock().await;

            // implicit heartbeat
            state
                .heartbeats
                .insert(requester.replica_id.clone(), Instant::now());

            // A never-seen-before incarnation is a (re)join of the group --
            // relay it upstream so clusterd can correlate it with an
            // in-flight rebuild (self-initiated event correlation, plan
            // section 3.3.6 ruling 4).
            if !state.known_replicas.contains(&requester.replica_id) {
                state.known_replicas.insert(requester.replica_id.clone());
                let event = Lighthouse::push_event(
                    &mut state,
                    FaultEvent {
                        seq: 0,
                        group_rank: 0,
                        source: "replica_joined".to_string(),
                        replica_id: requester.replica_id.clone(),
                        group_name: group_of(&requester.replica_id),
                        error_class: "REPLICA_JOINED".to_string(),
                        error_code: String::new(),
                        window: "none".to_string(),
                        step: requester.step,
                        retry_count: 0,
                        detail: format!(
                            "replica {} (group {}) registered with lighthouse",
                            requester.replica_id, group_of(&requester.replica_id)
                        ),
                        timestamp_ms: now_ms(),
                    },
                );
                Lighthouse::spawn_event_push(self.opt.clusterd_addr.clone(), event);
            }

            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester.clone(),
                },
            );
            let rx = state.channel.subscribe();

            // proactively run quorum tick
            self.clone()
                ._quorum_tick(&mut state)
                .map_err(|e| Status::from_error(e.into()))?;

            rx
        };

        let quorum = loop {
            let current_quorum = rx.recv().await.map_err(|e| Status::from_error(e.into()))?;

            if current_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == requester.replica_id)
            {
                break current_quorum;
            }

            // Only continue the loop if the replica is not in the quorum
            let mut state = self.state.lock().await;
            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester.clone(),
                },
            );
            info!("Replica {} not in quorum, retrying", &requester.replica_id);
        };

        let reply = LighthouseQuorumResponse {
            quorum: Some(quorum),
        };

        Ok(Response::new(reply))
    }

    async fn heartbeat(
        &self,
        request: Request<LighthouseHeartbeatRequest>,
    ) -> Result<Response<LighthouseHeartbeatResponse>, Status> {
        let replica_id = request.into_inner().replica_id;

        {
            let mut state = self.state.lock().await;
            state.heartbeats.insert(replica_id, Instant::now());
        }

        let reply = LighthouseHeartbeatResponse {};
        Ok(Response::new(reply))
    }
}

#[tonic::async_trait]
impl LighthouseControlService for Arc<Lighthouse> {
    async fn report_event(
        &self,
        request: Request<ReportEventRequest>,
    ) -> Result<Response<ReportEventResponse>, Status> {
        let event = request
            .into_inner()
            .event
            .ok_or_else(|| Status::invalid_argument("missing event"))?;

        let mut state = self.state.lock().await;
        // Fill the group name from the replica_id when the reporter left
        // it empty (lighthouse-side context is authoritative).
        let mut event = event;
        if event.group_name.is_empty() && !event.replica_id.is_empty() {
            event.group_name = group_of(&event.replica_id);
        }
        truncate_detail_tail(&mut event.detail, EVENT_DETAIL_MAX_CHARS);
        info!(
            "buffering fault event seq->{} source={} group={} class={} step={}",
            state.next_event_seq, event.source, event.group_name, event.error_class, event.step
        );
        let event = Lighthouse::push_event(&mut state, event);
        // Push-ONLY transport (v5.13): the buffered event also goes to
        // clusterd's sink fire-and-forget. Push failure only warns inside
        // the spawned task -- ReportEvent always succeeds (the 15s resend
        // sweep redelivers anything the push drops).
        Lighthouse::spawn_event_push(self.opt.clusterd_addr.clone(), event);

        Ok(Response::new(ReportEventResponse {}))
    }

    async fn get_observation(
        &self,
        _request: Request<ObservationRequest>,
    ) -> Result<Response<ObservationResponse>, Status> {
        let now = Instant::now();
        let state = self.state.lock().await;

        let max_step = state
            .heartbeats
            .keys()
            .filter_map(|replica_id| state.participants.get(replica_id))
            .map(|details| details.member.step)
            .chain(
                state
                    .prev_quorum
                    .as_ref()
                    .map(|q| q.participants.iter().map(|p| p.step).collect::<Vec<i64>>())
                    .unwrap_or_default(),
            )
            .max()
            .unwrap_or(-1);

        let mut replicas: Vec<ReplicaObservation> = Vec::new();
        let in_quorum_groups: HashSet<String> = state
            .prev_quorum
            .as_ref()
            .map(|q| q.participants.iter().map(|p| group_of(&p.replica_id)).collect())
            .unwrap_or_default();

        let mut seen_groups: HashSet<String> = HashSet::new();
        // Participants known to lighthouse (most recent incarnation per group)
        for (replica_id, details) in &state.participants {
            let group = group_of(replica_id);
            if !seen_groups.insert(group.clone()) {
                continue;
            }
            let heartbeat_age_ms = state
                .heartbeats
                .get(replica_id)
                .map(|h| now.duration_since(*h).as_millis() as i64)
                .unwrap_or(-1);
            replicas.push(ReplicaObservation {
                replica_id: replica_id.clone(),
                group_name: group.clone(),
                address: details.member.address.clone(),
                in_quorum: in_quorum_groups.contains(&group),
                heartbeating: heartbeat_age_ms >= 0
                    && heartbeat_age_ms < self.opt.heartbeat_timeout_ms as i64,
                excluded: state.excluded.contains(&group),
                step: details.member.step,
                heartbeat_age_ms,
            });
        }
        // Groups from the last quorum that no longer participate at all
        if let Some(prev) = &state.prev_quorum {
            for member in &prev.participants {
                let group = group_of(&member.replica_id);
                if seen_groups.insert(group.clone()) {
                    let heartbeat_age_ms = state
                        .heartbeats
                        .get(&member.replica_id)
                        .map(|h| now.duration_since(*h).as_millis() as i64)
                        .unwrap_or(-1);
                    replicas.push(ReplicaObservation {
                        replica_id: member.replica_id.clone(),
                        group_name: group.clone(),
                        address: member.address.clone(),
                        in_quorum: false,
                        // not heartbeating is a fact derived from the
                        // heartbeat age, not a constant: a replica can be
                        // outside the pending participant set (quorum
                        // formation clears it) and still fully alive
                        heartbeating: heartbeat_age_ms >= 0
                            && heartbeat_age_ms < self.opt.heartbeat_timeout_ms as i64,
                        excluded: state.excluded.contains(&group),
                        step: member.step,
                        heartbeat_age_ms,
                    });
                }
            }
        }
        // Replicas with a fresh heartbeat but no pending registration.
        // Quorum formation (and member-list churn) clears the pending
        // participant set; between that and each replica's next quorum()
        // call the participants map says nothing about liveness. Without
        // this block the observation reports a fully-heartbeating fleet as
        // gone, which clusterd's all-replicas-failed rule would act on.
        for (replica_id, last) in &state.heartbeats {
            let group = group_of(replica_id);
            if seen_groups.contains(&group) {
                continue;
            }
            let heartbeat_age_ms = now.duration_since(*last).as_millis() as i64;
            if heartbeat_age_ms >= self.opt.heartbeat_timeout_ms as i64 {
                continue;
            }
            seen_groups.insert(group.clone());
            replicas.push(ReplicaObservation {
                replica_id: replica_id.clone(),
                group_name: group.clone(),
                address: String::new(),
                in_quorum: in_quorum_groups.contains(&group),
                heartbeating: true,
                excluded: state.excluded.contains(&group),
                step: -1,
                heartbeat_age_ms,
            });
        }

        Ok(Response::new(ObservationResponse {
            quorum_id: state.quorum_id,
            max_step: max_step,
            replicas,
        }))
    }

    async fn send_command(
        &self,
        request: Request<CommandRequest>,
    ) -> Result<Response<CommandResponse>, Status> {
        let command = request.into_inner();
        let target = group_of(&command.target_replica);
        let command_id = command.command_id.clone();

        let addr = {
            let state = self.state.lock().await;
            Lighthouse::manager_address_for(&state, &target)
        };

        let addr = match addr {
            Some(addr) if !addr.is_empty() => addr,
            _ => {
                warn!(
                    "command {} target group {} has no known ManagerServer address",
                    command_id, target
                );
                return Ok(Response::new(CommandResponse {
                    accepted: false,
                    command_id,
                    detail: format!("no ManagerServer address known for group {}", target),
                    acked_ms: now_ms(),
                }));
            }
        };

        let connect_timeout = Duration::from_secs(5);
        let call_timeout = Duration::from_secs(10);
        let addr_for_log = addr.clone();

        let result = async move {
            let mut client =
                manager_control_client_new(addr, connect_timeout)
                .await
                .map_err(|e| Status::from_error(e.into()))?;
            let mut req = Request::new(ManagerCommandRequest {
                command: Some(command),
            });
            req.set_timeout(call_timeout);
            client.command(req).await
        }
        .await;

        match result {
            Ok(resp) => {
                let ack = resp.into_inner().ack.unwrap_or(CommandResponse {
                    accepted: false,
                    command_id: command_id.clone(),
                    detail: "empty ack".to_string(),
                    acked_ms: now_ms(),
                });
                info!(
                    "command {} forwarded to group {} at {}: accepted={} detail={}",
                    command_id, target, addr_for_log, ack.accepted, ack.detail
                );
                Ok(Response::new(ack))
            }
            Err(e) => {
                warn!(
                    "command {} forward to group {} at {} failed: {}",
                    command_id, target, addr_for_log, e
                );
                Ok(Response::new(CommandResponse {
                    accepted: false,
                    command_id,
                    detail: format!("forward failed: {}", e),
                    acked_ms: now_ms(),
                }))
            }
        }
    }

    async fn set_member_list(
        &self,
        request: Request<MemberInstructionRequest>,
    ) -> Result<Response<MemberInstructionResponse>, Status> {
        let req = request.into_inner();
        let instruction_id = req.instruction_id.clone();

        let mut state = self.state.lock().await;

        // Idempotency: re-delivery of the same instruction is a no-op
        // (MC-9: repeated instruction with the same id must not re-apply).
        if state.applied_instructions.contains(&instruction_id) {
            return Ok(Response::new(MemberInstructionResponse {
                accepted: true,
                detail: "duplicate instruction ignored".to_string(),
                quorum_id: state.quorum_id,
            }));
        }

        let mut changed = false;
        for group in req.exclude {
            if !group.is_empty() && state.excluded.insert(group.clone()) {
                info!("clusterd excluded group {} from quorum", group);
                changed = true;
            }
        }
        for group in req.restore {
            if state.excluded.remove(&group) {
                info!("clusterd restored group {} to quorum", group);
                changed = true;
            }
        }

        state.applied_instructions.insert(instruction_id.clone());
        if state.applied_instructions.len() > 1024 {
            // bound the idempotency memory; old instructions are unlikely to
            // be re-delivered by then
            if let Some(first) = state.applied_instructions.iter().next().cloned() {
                state.applied_instructions.remove(&first);
            }
        }

        if changed {
            // Do NOT clear state.participants here. Managers park inside
            // their quorum() RPC waiter loop and only re-register when a
            // broadcast wakes them; a clear with no quorum to broadcast
            // leaves them parked with their registration deleted and the
            // quorum can never form again (observed: a healthy booting job
            // died with a 300s quorum timeout exactly when clusterd's
            // restore instruction landed). Exclusion/restore are enforced
            // live by quorum_compute's candidate filter, so the pending
            // registrations can stay.
            let excluded_snapshot: Vec<String> =
                state.excluded.iter().cloned().collect();
            let event = Lighthouse::push_event(
                &mut state,
                FaultEvent {
                    seq: 0,
                    group_rank: 0,
                    source: "quorum_change".to_string(),
                    replica_id: String::new(),
                    group_name: String::new(),
                    error_class: "MEMBER_INSTRUCTION".to_string(),
                    error_code: String::new(),
                    window: "none".to_string(),
                    step: -1,
                    retry_count: 0,
                    detail: format!(
                        "member instruction {} applied, excluded={:?}",
                        instruction_id, excluded_snapshot
                    ),
                    timestamp_ms: now_ms(),
                },
            );
            Lighthouse::spawn_event_push(self.opt.clusterd_addr.clone(), event);
            self.clone()
                ._quorum_tick(&mut state)
                .map_err(|e| Status::from_error(e.into()))?;
        }

        Ok(Response::new(MemberInstructionResponse {
            accepted: true,
            detail: "ok".to_string(),
            quorum_id: state.quorum_id,
        }))
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

#[derive(Template)]
#[template(path = "status.html")]
struct StatusTemplate {
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
    quorum_status: String,
    num_participants: i64,
    max_step: i64,
    heartbeats: HashMap<String, Instant>,

    // visualization thresholds
    old_age_threshold: Instant,
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Sub;

    use tonic::transport::Channel;

    use super::*;
    use crate::net::connect;
    use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;

    async fn lighthouse_client_new(addr: String) -> Result<LighthouseServiceClient<Channel>> {
        let conn = connect(addr, Duration::from_secs(10)).await?;
        Ok(LighthouseServiceClient::new(conn))
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            events: VecDeque::new(),
            next_event_seq: 1,
            event_acked: HashSet::new(),
            excluded: HashSet::new(),
            known_replicas: HashSet::new(),
            last_alive: HashSet::new(),
            applied_instructions: HashSet::new(),
        };

        let now = Instant::now();

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("New quorum not ready, only have 0 participants, need min_replicas 1 [0/0 participants healthy]"), "{}", reason);

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        // all healthy workers participating
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        // add healthy worker but not participating
        state.heartbeats.insert("c".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("join timeout"), "{}", reason);

        // increase elapsed time to pass join timeout
        state.participants.get_mut("a").unwrap().joined =
            now.sub(Duration::from_secs(10 * 60 * 60));
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_heartbeats() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            events: VecDeque::new(),
            next_event_seq: 1,
            event_acked: HashSet::new(),
            excluded: HashSet::new(),
            known_replicas: HashSet::new(),
            last_alive: HashSet::new(),
            applied_instructions: HashSet::new(),
        };

        let now = Instant::now();

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        assert!(
            reason.contains("[1/1 participants healthy][1 heartbeating]"),
            "{}",
            reason
        );

        // expired heartbeat
        state
            .heartbeats
            .insert("a".to_string(), now.sub(Duration::from_secs(10)));

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(
            reason.contains("[0/1 participants healthy][0 heartbeating]"),
            "{}",
            reason
        );

        // 1 healthy, 1 expired
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            events: VecDeque::new(),
            next_event_seq: 1,
            event_acked: HashSet::new(),
            excluded: HashSet::new(),
            known_replicas: HashSet::new(),
            last_alive: HashSet::new(),
            applied_instructions: HashSet::new(),
        };

        let now = Instant::now();

        assert!(!quorum_compute(now, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        // Not proceeding since one worker is alive but not participating
        state.heartbeats.insert("b".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("need at least half"), "{}", reason);

        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![QuorumMember {
                replica_id: "a".to_string(),
                address: "".to_string(),
                store_address: "".to_string(),
                step: 1,
                world_size: 1,
                shrink_only: false,
                data: String::new(),
                commit_failures: 0,
            }],
            created: Some(SystemTime::now().into()),
        });

        assert!(quorum_compute(now, &state, &opt).0.is_some());

        // test expanding quorum w/ fast quorum
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_shrink_only() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            events: VecDeque::new(),
            next_event_seq: 1,
            event_acked: HashSet::new(),
            excluded: HashSet::new(),
            known_replicas: HashSet::new(),
            last_alive: HashSet::new(),
            applied_instructions: HashSet::new(),
        };

        let now = Instant::now();

        state.prev_quorum = Some(Quorum {
            quorum_id: 1,
            participants: vec![
                QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
                QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            ],
            created: Some(SystemTime::now().into()),
        });

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: true,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);

        // insert particpant that was not in prev quorum
        state.participants.insert(
            "c".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "c".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: true,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("c".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        assert!(reason.contains("[shrink_only=true]",), "{}", reason);

        let quorum = quorum_met.unwrap();
        assert!(quorum.len() == 1);
        assert!(quorum[0].replica_id == "a");

        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_e2e() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };
        let lighthouse = Lighthouse::new(opt).await?;

        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await.unwrap();

        {
            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: "foo".to_string(),
            });

            let _response = client.heartbeat(request).await.unwrap();
        }

        {
            let request = tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(QuorumMember {
                    replica_id: "foo".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 10,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                }),
            });

            let response = client.quorum(request).await.unwrap();
            let quorum = response.into_inner().quorum.unwrap();
            assert_eq!(quorum.participants.len(), 1);
        }

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_split_brain() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = State {
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
            heartbeats: HashMap::new(),
            events: VecDeque::new(),
            next_event_seq: 1,
            event_acked: HashSet::new(),
            excluded: HashSet::new(),
            known_replicas: HashSet::new(),
            last_alive: HashSet::new(),
            applied_instructions: HashSet::new(),
        };

        let now = Instant::now();

        assert!(!quorum_compute(now, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                    shrink_only: false,
                    data: String::new(),
                    commit_failures: 0,
                },
            },
        );
        state.heartbeats.insert("a".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);

        // Not proceeding since one worker is alive but not participating
        state.heartbeats.insert("b".to_string(), now);
        let (quorum_met, reason) = quorum_compute(now, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);
        assert!(reason.contains("New quorum not ready, only have 1 participants, need at least half of 2 healthy workers [1/1 participants healthy][2 heartbeating]"), "{}", reason);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_changed() {
        let a = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];
        let b = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "changed".to_string(),
            store_address: "changed".to_string(),
            step: 1000,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];

        // replica_id is the same
        assert!(!quorum_changed(&a, &b));

        let c = vec![QuorumMember {
            replica_id: "2".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
            shrink_only: false,
            data: String::new(),
            commit_failures: 0,
        }];
        // replica_id changed
        assert!(quorum_changed(&a, &c));
    }

    #[tokio::test]
    async fn test_lighthouse_join_during_shrink() -> Result<()> {
        fn create_member(id: &str, addr_num: &str, step: i64, shrink_only: bool) -> QuorumMember {
            QuorumMember {
                replica_id: id.to_string(),
                address: format!("addr{}", addr_num),
                store_address: format!("store{}", addr_num),
                step,
                world_size: 1,
                shrink_only,
                data: String::new(),
                commit_failures: 0,
            }
        }

        fn create_request(member: &QuorumMember) -> tonic::Request<LighthouseQuorumRequest> {
            tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(member.clone()),
            })
        }

        let opt = LighthouseOpt {
            min_replicas: 2,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        // Start the lighthouse service
        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // Create client to interact with lighthouse
        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        // 1. First quorum
        let mut first_request = create_request(&create_member("replica0", "1", 1, false));
        let mut second_request = create_request(&create_member("replica1", "2", 1, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let first_response = client.quorum(second_request).await?;
        let first_quorum = first_response.into_inner().quorum.unwrap();
        assert_eq!(first_quorum.participants.len(), 2);
        assert_eq!(first_quorum.participants[0].replica_id, "replica0");
        assert_eq!(first_quorum.participants[1].replica_id, "replica1");
        assert_eq!(first_quorum.participants[1].step, 1);

        // 2. Quorum without joiner
        let joining_request = create_request(&create_member("joiner", "3", 1, false));
        let joining_task = tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(joining_request).await }
        });

        // Try to shrink only
        first_request = create_request(&create_member("replica0", "1", 2, true));
        second_request = create_request(&create_member("replica1", "2", 2, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let second_response = client.quorum(second_request).await?;
        let second_quorum = second_response.into_inner().quorum.unwrap();
        assert!(
            second_quorum
                .participants
                .iter()
                .all(|p| p.replica_id != "joiner")
        );
        assert_eq!(second_quorum.participants.len(), 2);
        assert_eq!(second_quorum.participants[0].replica_id, "replica0");
        assert_eq!(second_quorum.participants[1].replica_id, "replica1");
        assert_eq!(second_quorum.participants[1].step, 2);

        // 3. Quorum with joiner
        first_request = create_request(&create_member("replica0", "1", 3, false));
        second_request = create_request(&create_member("replica1", "2", 3, false));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let third_response = client.quorum(second_request).await?;
        let third_quorum = third_response.into_inner().quorum.unwrap();
        assert!(
            third_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == "joiner")
        );
        assert_eq!(third_quorum.participants.len(), 3);
        assert_eq!(third_quorum.participants[2].step, 3);

        let join_result = joining_task.await?;
        let join_quorum = join_result.unwrap().into_inner().quorum.unwrap();
        assert!(
            join_quorum
                .participants
                .iter()
                .any(|p| p.replica_id == "joiner")
        );
        assert_eq!(join_quorum.participants.len(), 3);
        assert_eq!(join_quorum.participants[2].step, 3);

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_commit_failures() -> Result<()> {
        fn create_member(id: &str, commit_failures: i64) -> QuorumMember {
            QuorumMember {
                replica_id: id.to_string(),
                address: format!("addr{}", id),
                store_address: format!("store{}", id),
                step: 10,
                world_size: 1,
                shrink_only: false,
                data: String::new(),
                commit_failures,
            }
        }

        fn create_request(member: &QuorumMember) -> tonic::Request<LighthouseQuorumRequest> {
            tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(member.clone()),
            })
        }

        let opt = LighthouseOpt {
            min_replicas: 2,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1000,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        // Start the lighthouse service
        let lighthouse = Lighthouse::new(opt).await?;
        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        // Create client to interact with lighthouse
        let mut client = lighthouse_client_new(lighthouse.address()).await?;

        // First two quorums should be stable
        for _i in 0..2 {
            let first_request = create_request(&create_member("replica0", 0));
            let second_request = create_request(&create_member("replica1", 0));

            tokio::spawn({
                let mut client = client.clone();
                async move { client.quorum(first_request).await }
            });
            let first_response = client.quorum(second_request).await?;
            let first_quorum = first_response.into_inner().quorum.unwrap();
            assert_eq!(first_quorum.quorum_id, 1);
            assert_eq!(first_quorum.participants.len(), 2);
            assert_eq!(first_quorum.participants[0].commit_failures, 0);
            assert_eq!(first_quorum.participants[1].commit_failures, 0);
        }

        // commit_failures should increment quorum_id
        let first_request = create_request(&create_member("replica0", 0));
        let second_request = create_request(&create_member("replica1", 2));

        tokio::spawn({
            let mut client = client.clone();
            async move { client.quorum(first_request).await }
        });
        let first_response = client.quorum(second_request).await?;
        let first_quorum = first_response.into_inner().quorum.unwrap();
        assert_eq!(first_quorum.quorum_id, 2);
        assert_eq!(first_quorum.participants.len(), 2);
        assert_eq!(first_quorum.participants[0].commit_failures, 0);
        assert_eq!(first_quorum.participants[1].commit_failures, 2);

        lighthouse_task.abort();
        Ok(())
    }
}
