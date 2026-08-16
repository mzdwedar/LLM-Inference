import json
import random
from pathlib import Path

# Resolve paths against the repo root so this runs from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# 1. EXTENDED ENTERPRISE MODULES (Bulkier, multi-section markdown with metadata)
# ==============================================================================
long_contexts = {
    "HR_GERMANY": (
        "# Document Ref: DE-HR-2026-V4\n"
        "Classification: Internal Use Only\n"
        "Authors: People Operations Compliance Group Europe\n"
        "Last Modified: June 12, 2026\n"
        "--------------------------------------------------\n\n"
        "## Executive Summary\n"
        "This document governs the compliance, legal boundaries, and equipment allocation policies for "
        "all personnel operating under the jurisdiction of the Federal Republic of Germany. All personnel "
        "must adhere strictly to these guidelines to ensure corporate tax neutrality and physical workplace safety.\n\n"
        "## Section 1.0: Architectural Compliance and German Labor Regulations\n"
        "All employees registered within the Federal Republic of Germany are governed under the localized "
        "remote work framework agreement (Betriebsvereinbarung Telearbeit). The policy distinguishes strictly "
        "between 'Mobiles Arbeiten' (mobile working) and 'Telearbeit' (dedicated home-office teleworking).\n\n"
        "Under the current mandate, employees affiliated with corporate hubs in Berlin, Munich, or Frankfurt "
        "are permitted a maximum allocation of 180 fluid remote-working calendar days per fiscal year. "
        "Core presence windows are strictly instituted across all engineering and infrastructure teams "
        "between 10:00 and 16:00 Central European Time (CET) to guarantee synchronous block scheduling across distributed "
        "architectural development projects. Daily rest period regulations mandate an uninterrupted 11-hour block "
        "of offline time between operational shifts, governed automatically by centralized Okta platform access logs.\n\n"
        "Relocation of a primary residency or persistent remote network node outside the official bounds of Germany "
        "triggers instant tax re-classification. Any cross-border remote work scenario requires a formal HR processing "
        "window of 45 business days to clear mandatory regulatory hurdles, secure localized health insurance riders, "
        "and process the statutory A1 Certificate for European Union labor postings. Failure to report geo-location "
        "modifications within the corporate payroll system constitutes a direct compliance breach and will result "
        "in disciplinary escalation.\n\n"
        "## Section 2.0: Digital Peripherals and Infrastructure Provisioning\n"
        "The enterprise provides localized hardware configurations consisting of a standard corporate laptop asset, "
        "encrypted hardware security keys, and automated access provisioning via our centralized Okta directory. "
        "Home office installations classified under full Telearbeit are eligible for ergonomic equipment lifecycle "
        "tracking via the central procurement framework. Budget tracking codes must be approved by the designated "
        "cost-center manager prior to asset acquisition through the approved internal portal vendor ecosystem.\n\n"
        "### Appendix A: Equipment Lifecycle Matrix\n"
        "| Asset Code | Description | Approved Vendor | Lifecycle | Cost Center Code |\n"
        "|---|---|---|---|---|\n"
        "| LP-X1-GER | Carbon Enterprise Laptop | Lenovo DE | 36 Months | CC-ENG-INFRA |\n"
        "| MON-4K-32 | UltraWide 32-inch Monitor | Dell Direct | 48 Months | CC-ENG-INFRA |\n"
        "| HW-KEY-YUB | Cryptographic Security Key | Yubico EU | 24 Months | CC-SEC-OPS |\n"
        "| CH-ERG-01 | Ergonomic Task Chair | Herman Miller | 60 Months | CC-GLOBAL-HR |\n\n"
        "## Section 3.0: Employee Wellness and Work-Life Integration\n"
        "The enterprise remains committed to preventing digital fatigue. Managers are instructed to monitor "
        "communication channels (Slack, Email) to ensure team members are not sending non-urgent messages outside "
        "of regional core presence hours. Virtual tea breaks and asynchronous check-ins are highly encouraged "
        "to maintain strong interpersonal connections across our remote engineering organization."
    ),
    "INFRA_MLOPS": (
        "# Document Ref: INFRA-MLOPS-GPU-2026-V12\n"
        "Classification: Confidential - Engineering Teams Only\n"
        "Authors: Core Machine Learning Platform Architecture Group\n"
        "Last Modified: July 02, 2026\n"
        "--------------------------------------------------\n\n"
        "## Executive Summary\n"
        "This architectural runbook details the engineering pathways for transitioning production, high-throughput "
        "large language model (LLM) inference workloads from legacy Triton clusters to heterogeneous hardware "
        "profiles including Apple Silicon testbeds and high-performance Slurm-orchestrated GPU environments.\n\n"
        "## Section 9.2: Distributed Inference Migration and Kernel Compile Matrix\n"
        "Engineering teams transitioning production workloads from dedicated NVIDIA Triton clusters to local "
        "Apple Silicon development machines (specifically M3 Max/Ultra 128GB unmanaged nodes) or remote multi-GPU "
        "Slurm partitions must enforce matching tensor memory layouts to prevent downstream model divergence.\n\n"
        "When executing high-throughput inference simulations utilizing vLLM backends, the generation pipeline "
        "relies on Automatic Prefix Caching (APC) to tokenize and hash system contexts in blocks of 16 tokens using "
        "cryptographic xxHash64 lookups on the GPU. If a workload utilizes mixed-precision FP8 KV-caches (specifically "
        "utilizing the e4m3 precision format), memory allocations are cut exactly by half compared to standard BF16 layouts, "
        "allowing larger internal batch configurations without causing out-of-memory (OOM) faults during heavy prefill steps.\n\n"
        "However, local local-loop testing on M3 Silicon lacks native FP8 tensor core accumulation units; therefore, "
        "local verification nodes must fallback to simulated INT8 or half-precision (FP16) quantization matrices via "
        "localized MLX compilations. For distributed execution on Slurm clusters running H100 or HGX architectures, "
        "chunked prefill configuration flags (`--enable-chunked-prefill=True`) must be activated to process long input "
        "payloads concurrently alongside active token generation streams. This prevents severe spikes in Time-to-First-Token "
        "(TTFT) metrics when concurrent user traffic bursts according to non-uniform Poisson arrival rates.\n\n"
        "Any custom pipeline lacking valid tensor shape mapping inside the Triton configuration repository "
        "will fail compilation checks during CI/CD orchestration. Standard model signatures must be declared in "
        "config.pbtxt with appropriate dimension bounds.\n\n"
        "### Appendix B: Memory Layout & Performance Reference Matrix\n"
        "| Hardware Architecture | Target Kernel | Recommended Cache Format | Chunked Prefill | Target TTFT (p95) |\n"
        "|---|---|---|---|---|\n"
        "| Apple M3 Max 128GB | MLX Custom Compile | FP16 / INT8 Simulated | Unsupported | < 450ms |\n"
        "| NVIDIA H100 HGX | Triton TensorRT-LLM | FP8 (e4m3 Format) | Required | < 120ms |\n"
        "| NVIDIA A100 PCIe | Triton Python / vLLM | BF16 Standard | Recommended | < 280ms |\n\n"
        "## Section 9.3: Automated Failover and Load-Balancing Topologies\n"
        "To guarantee high availability (99.99% uptime), production gateways are fronted by dynamic round-robin "
        "routing layers. In the event of a cluster-wide CUDA driver exception, the gateway automatically sheds "
        "non-critical analytical workloads to auxiliary cold-standby GPU pools, reserving the primary H100 "
        "fabric exclusively for real-time transactional user queries."
    ),
    "SECURITY_INFOSEC": (
        "# Document Ref: SEC-INFOSEC-OIDC-2026-V2\n"
        "Classification: Strict Cryptographic Security Clearance Required\n"
        "Authors: Enterprise Identity and Access Management Group\n"
        "Last Modified: May 19, 2026\n"
        "--------------------------------------------------\n\n"
        "## Executive Summary\n"
        "This reference manual establishes the security controls, access management lifecycles, and data loss "
        "prevention mechanisms across all physical and virtual enterprise nodes. Strict enforcement of zero-trust "
        "architectures is mandatory across all cloud environments.\n\n"
        "## Section 12.4: Infosec Protocols, OIDC Token Lifecycles, and Data Exfiltration Preventions\n"
        "The architecture governing internal platform accessibility relies on strict OpenID Connect (OIDC) "
        "tokens managed via centralized Okta identity pools. Session lifecycles for high-clearance production "
        "engineers are bound to an absolute ceiling of 8 continuous hours, after which cryptographic rotation "
        "mandates a silent re-authentication handshake.\n\n"
        "If an active session is detected jumping across discrete geographic routing nodes—such as a network "
        "transition from a residential German ISP to an external corporate VPN node within a short window—the security "
        "automated response network (SARN) triggers an immediate step-up challenge requiring a physical WebAuthn "
        "hardware key verification (such as a corporate YubiKey).\n\n"
        "Data access patterns are monitored for anomalies; pulling raw files or documentation structures that "
        "exceed a combined weight of 250 Megabytes within a trailing 15-minute window flags the identity payload as "
        "a potential data exfiltration risk. For distributed development clusters and high-performance engineering "
        "groups operating remote Jupyter environments or Slurm head-nodes, API access keys must never be committed "
        "to repository states.\n\n"
        "Any secret string matching a high-entropy regex fingerprint that is detected during automated pre-commit "
        "hook scanning will cause an immediate branch lock and notify the internal security audit division. "
        "Exceptions for specialized automated workflow pipelines (e.g., Prefect automated orchestration runtimes) "
        "are granted exclusively through short-lived IAM roles using AWS STS token generation with explicit boundaries.\n\n"
        "### Appendix C: Security Event Escalation Tiers\n"
        "| Incident Tier | Trigger Condition | Automated Action | SLA for Human Review |\n"
        "|---|---|---|---|\n"
        "| Tier 1: Low | > 100MB data download in 15m | Log event & soft alert | 24 Hours |\n"
        "| Tier 2: Medium | Multi-country IP jump | Temporary access suspension | 1 Hour |\n"
        "| Tier 3: High | Secret key leak / raw exfiltrate | Hard lock, credential revocation | Immediate (10 mins) |\n\n"
        "## Section 12.5: Physical Asset Disposal and Offboarding Protocols\n"
        "Upon termination of an employment contract, all physical cryptographic assets, including corporate security keys "
        "and physical computing units, must be mailed back to regional IT hubs via secure, tracked courier service "
        "within 3 business days. Absolute sanitization of local storage drives must be verified before recycling."
    )
}

# ==============================================================================
# 2. ADDITIONAL DISTRACTOR BLOCKS (Used to artificially pad prompt lengths)
# ==============================================================================
distractor_pool = [
    (
        "## Distractor Section A: Legacy Travel Policy Guidelines (Archive 2024)\n"
        "Travel expenses must be logged within 14 business days of returning from international operations. "
        "Class of travel is strictly limited to economy class for flights under 6 hours in duration. "
        "Meal allocations must not exceed 50 EUR per diem in metropolitan hubs. Standard hotel bookings "
        "should utilize partnered chains listed inside the global concur portal interface. Out-of-pocket expenses "
        "under 15 EUR do not require physical receipt scans but require written business rationale."
    ),
    (
        "## Distractor Section B: Core Values and Company Mission Manifesto\n"
        "We are dedicated to building a culture of radical ownership, open communication, and cross-functional "
        "excellence. Our teams work at the boundary of what is possible in distributed computing and localized "
        "artificial intelligence. We strive for engineering elegance in every line of code we write, prioritizing "
        "safety, security, scalability, and empathetic team relationships over rapid, unverified deployments."
    ),
    (
        "## Distractor Section C: Legacy Database Sharding Runbook (PostgreSQL 14)\n"
        "When managing older relational storage systems, verify that table partitions do not exceed "
        "fifty million active rows. Partition schemes should partition strictly on `created_at` timestamps "
        "on a monthly interval. Reindexing tasks must be scheduled during off-peak windows (02:00 to 04:00 UTC) "
        "to avoid query lock starvation. Ensure that autovacuum configurations are tuned aggressively to prevent "
        "wraparound transaction ID issues."
    )
]

# ==============================================================================
# 3. UNIQUE QUESTIONS BY DOMAIN
# ==============================================================================
questions = {
    "HR_GERMANY": [
        "What is our remote work policy in Germany?",
        "Do we get a home office budget?",
        "What happens if I move from Cologne to Aachen while working remotely?",
        "How many remote working days are allowed per year?",
        "What are the core presence hours for engineering teams?",
        "What happens if I work outside of German border topology?",
        "How long does it take to process cross-border remote work?",
        "What equipment can I order through the internal portal?"
    ],
    "INFRA_MLOPS": [
        "How does running FP8 KV-caches differ when testing locally on an M3 Max versus running in a production NVIDIA HGX Slurm cluster?",
        "What hashing mechanism does Automatic Prefix Caching use?",
        "Why should we enable chunked prefill on Slurm clusters?",
        "What block size does vLLM use to hash system contexts?",
        "What is the impact of e4m3 precision on memory allocations?",
        "How are local loops verified without native FP8 units?",
        "What triggers an infrastructure compilation check failure?"
    ],
    "SECURITY_INFOSEC": [
        "What security actions are taken if my login location changes unexpectedly?",
        "What happens if I download a large volume of documentation?",
        "What is the maximum lifecycle for an OIDC token session?",
        "What hardware is required if the security network triggers a step-up challenge?",
        "How are secret strings tracked during repository commits?",
        "How are API keys managed for Prefect workflow pipelines?"
    ]
}

# ==============================================================================
# 4. GENERATION ENGINE (Constructs long prompts with varying padding sizes)
# ==============================================================================
output_file = REPO_ROOT / "data" / "rag_workload.jsonl"
output_file.parent.mkdir(parents=True, exist_ok=True)
total_prompts = 2000

# Base formatting templates to simulate actual RAG instructions
system_instruction = (
    "System Instruction: You are an advanced enterprise QA assistant. "
    "Use the following retrieved corporate document sections to accurately and professionally "
    "answer the user query at the end. Do not assume or extrapolate facts beyond what is written. "
    "If the answer cannot be confidently verified in the context, explicitly state that the "
    "information is missing from current documentation.\n\n"
)

with open(output_file, "w", encoding="utf-8") as f:
    generated_count = 0
    
    while generated_count < total_prompts:
        # Pick our target domain
        domain = random.choice(list(long_contexts.keys()))
        target_context = long_contexts[domain]
        available_questions = questions[domain]
        
        # Determine burst size (forces heavy prefix cache reuse under exact same mock retrieval batch)
        burst_size = min(random.randint(3, 6), total_prompts - generated_count)
        
        # To simulate a realistic enterprise search result, we inject 2 distractors + target context.
        # This keeps the "prefix" structurally constant across the burst, allowing APC to cache it completely.
        retrieved_docs = [target_context] + distractor_pool
        random.shuffle(retrieved_docs)  # Shuffle so the target document is placed in different positions
        
        # Build the shared document block
        retrieved_block = "\n\n=== RETRIEVED DOCUMENT ===\n".join(retrieved_docs)
        
        # To make it even longer, we can duplicate parts of the distractor pool to bulk out the file to ~3,000+ words
        # (This translates to ~3,500 to 4,500 tokens, mirroring real long-context enterprise RAG).
        extra_bulk = "\n\n".join(random.sample(distractor_pool, 2))
        
        # Assemble the final shared context prefix
        shared_prefix = (
            f"{system_instruction}"
            f"=== RETRIEVED ENTERPRISE DOCUMENTATION SECTIONS ===\n"
            f"{retrieved_block}\n\n"
            f"=== SUPPLEMENTAL GUIDELINES ===\n"
            f"{extra_bulk}\n\n"
            f"==================================================\n"
        )
        
        for _ in range(burst_size):
            q = random.choice(available_questions)
            # Combine the long shared prefix with the unique question suffix
            full_prompt = f"{shared_prefix}User Query: {q}\nHelpful Assistant Answer:"
            
            row = {
                "prompt": full_prompt,
                "output_len": random.randint(100, 300)  # Standard corporate reply length
            }
            f.write(json.dumps(row) + "\n")
            generated_count += 1

print(f"Successfully generated {generated_count} long-form RAG simulation rows inside {output_file}")