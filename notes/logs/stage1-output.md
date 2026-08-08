# Output of !python 01_index.py


==========================================================================
[1] Load SciFact (BEIR) and sanity-check the counts
==========================================================================
100% 5183/5183 [00:00<00:00, 88108.42it/s]
  corpus docs         : 5183      (published: 5183)
  test queries        : 300       (published: 300)
  qrels (q,doc) pairs : 339
  distinct gold docs  : 283
  queries w/ >1 gold  : 23
  text field          : doc_text() = (title + " " + text).strip()
  docid order         : sorted numerically when all docids are digit strings, else lexicographically
  load time           : 0.2s
  -> counts match published BEIR SciFact stats

==========================================================================
[2] Token-length audit -> chunk policy
==========================================================================
  tokenizer           : BAAI/bge-base-en-v1.5 (with special tokens)
  encoder limit       : 512 tokens
  min / median / mean : 70 / 316 / 337.2
  p95 / p99 / max     : 567 / 760 / 1939
  docs over limit     : 455 (8.779%)
  audit time          : 5.9s

  chunk_policy        : chunked
  dedup_needed        : True
  reason              : 455 docs (8.78%) exceed 512 tokens, above the 1% tolerance -> chunk, and Stage 2 MUST collapse chunks to doc level before computing metrics against doc-level qrels
2026-08-06 04:34:41.666651: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  embedding units     : 5661

  docs over 512       : 455
  ...of which GOLD    : 46 of 283 gold docs (16.3%)
  qrels pairs at risk : 53 of 339
  -> chunked instead of truncated, so no gold evidence text is discarded

  docs actually split : 455
  chunks from those   : 933  (mean 2.05, max 5 per doc)
  -> !! Stage 2 MUST collapse chunks to doc level BEFORE metrics, or recall inflates

==========================================================================
[3] Embed passages (dense arm)
==========================================================================
  model               : BAAI/bge-base-en-v1.5
  device / dtype      : cuda / float16
  attn implementation : sdpa   (never flash_attention_2 on sm_75)
  batch size          : 64
  passage convention  : raw text, NO instruction/prefix (bge, not e5)
Batches: 100% 89/89 [00:30<00:00,  2.91it/s]

  model load          : 5.5s
  embed wall-clock    : 30.9s  (183 units/s)
  vectors             : shape=(5661, 768) dtype=float32
  normalization       : cast fp16->float32, then faiss.normalize_L2 in place (IP == cosine)
  L2 norm min/max     : 1.000000 / 1.000000  (expect ~1.0)

  peak VRAM allocated : 0.82 GiB
  peak VRAM reserved  : 1.07 GiB   (ceiling 14.5 GiB)
  -> well under the ceiling

==========================================================================
[4] Build + persist FAISS IndexFlatIP
==========================================================================
  index type          : IndexFlatIP (exact, exhaustive; metric=inner product)
  ntotal              : 5661
  dim (index.d)       : 768
  build+write         : 0.04s
  file                : /content/drive/MyDrive/slotA-rag-harness/artifacts/index/faiss.index  (16.6 MiB)
  -> ntotal == embedded units (5661) OK

==========================================================================
[5] Build + persist BM25 (sparse arm)
==========================================================================
  implementation      : rank_bm25.BM25Okapi (k1/b library defaults)
  tokenizer           : str.lower().split() — no stemming, no stopword removal, punctuation retained
  granularity         : doc-level, unchunked, untruncated (full text)
  docs                : 5183
  avg tokens/doc      : 214.6
  vocabulary size     : 87379
  build+pickle        : 0.71s
  file                : /content/drive/MyDrive/slotA-rag-harness/artifacts/index/bm25.pkl  (8.8 MiB)

==========================================================================
[6] Write docmap.json (ordinal <-> docid <-> text)
==========================================================================
  ordinals            : 5661
  docids              : 5183
  file                : /content/drive/MyDrive/slotA-rag-harness/artifacts/index/docmap.json  (15.3 MiB)
  -> every ordinal round-trips through docid_to_ordinals OK

==========================================================================
Stage 1 complete
==========================================================================
  chunk_policy   : chunked  (dedup_needed=True)
  faiss ntotal   : 5661   dim: 768
  bm25 docs      : 5183
  embed time     : 30.9s
  peak VRAM      : 1.07 GiB reserved / 0.82 GiB allocated
  TOTAL          : 55.6s

  meta written   : /content/drive/MyDrive/slotA-rag-harness/artifacts/index/index_meta.json

  Next: python verify_stage1.py   (fresh-process reload + round-trip + smoke)


# Output of !python verify_stage.py



==========================================================================
[3] Cold reload from disk (fresh process)
==========================================================================
  faiss.read_index    : OK  ntotal=5661  dim=768  type=IndexFlatIP
  bm25 unpickle       : OK  docs=5183  rank_bm25=0.2.2
  docmap.json         : OK  n_units=5661  policy=chunked  dedup_needed=True
  index_meta.json     : OK  built 2026-08-06T04:35:30+00:00
      faiss.index          16.59 MiB
      bm25.pkl              8.82 MiB
      docmap.json          15.29 MiB
      index_meta.json       0.00 MiB

==========================================================================
[1] Corpus counts (re-read from the cached BEIR download)
==========================================================================
100% 5183/5183 [00:00<00:00, 95545.89it/s]
  corpus docs         : 5183  (published 5183)
  test queries        : 300   (published 300)
  qrels pairs         : 339
  -> matches published stats

==========================================================================
[2] FAISS shape assertions
==========================================================================
  index.ntotal        : 5661
  index.d             : 768  (expect 768 for bge-base)
  docmap n_units      : 5661
  meta n_units        : 5661
  OK    ntotal == docmap n_units
  OK    ntotal == meta n_units
  OK    ntotal == len(ordinal_to_docid)
  OK    index.d == 768
  OK    bm25 docs == corpus docs

==========================================================================
[4] Round-trip: ordinal <-> docid <-> text
==========================================================================
  multi-chunk docs    : 455; probing the deepest (5 chunks) and a 2-chunk doc
  probing docids      : ['4983', '16172576', '198309074', '10749308', '9967265']

  docid 4983  (1 chunk)
      ordinals            : [0]
      ordinal->docid back : ['4983']   OK
      bm25 row            : 0
      in BEIR corpus      : True
      text                : Microstructural development of human newborn cerebral white matter assessed in vivo by d...

  docid 16172576  (1 chunk)
      ordinals            : [2830]
      ordinal->docid back : ['16172576']   OK
      bm25 row            : 2595
      in BEIR corpus      : True
      text                : Inter- and Intra-Host Viral Diversity in a Large Seasonal DENV2 Outbreak BACKGROUND High...

  docid 198309074  (1 chunk)
      ordinals            : [5660]
      ordinal->docid back : ['198309074']   OK
      bm25 row            : 5182
      in BEIR corpus      : True
      text                : Adhesion molecules and chemokines: relation to anthropometric, body composition, biochem...

  docid 10749308  (5 chunks)
      ordinals            : [1987, 1988, 1989, 1990, 1991]
      ordinal->docid back : ['10749308', '10749308', '10749308', '10749308', '10749308']   OK
      ordinals contiguous : True   OK
      bm25 row            : 1834
      in BEIR corpus      : True
      text                : Placebo-Controlled Trials and Active-Control Trials in the Evaluation of New Treatments....

  docid 9967265  (2 chunks)
      ordinals            : [1860, 1861]
      ordinal->docid back : ['9967265', '9967265']   OK
      ordinals contiguous : True   OK
      bm25 row            : 1724
      in BEIR corpus      : True
      text                : Surgical versus medical treatment with cyclooxygenase inhibitors for symptomatic patent ...

2026-08-06 04:36:50.263641: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  Alignment proof — re-embed the exact unit text and compare to the stored vector:
      ordinal 0     docid 4983       cos(stored, re-embedded) = 1.000000  OK
      ordinal 2830  docid 16172576   cos(stored, re-embedded) = 1.000000  OK
      ordinal 5660  docid 198309074  cos(stored, re-embedded) = 1.000000  OK
      ordinal 1987  docid 10749308   cos(stored, re-embedded) = 1.000000  OK
      ordinal 1988  docid 10749308   cos(stored, re-embedded) = 1.000000  OK
      ordinal 1989  docid 10749308   cos(stored, re-embedded) = 1.000000  OK
      ordinal 1990  docid 10749308   cos(stored, re-embedded) = 1.000000  OK
      ordinal 1991  docid 10749308   cos(stored, re-embedded) = 1.000000  OK
      ordinal 1860  docid 9967265    cos(stored, re-embedded) = 1.000000  OK
      ordinal 1861  docid 9967265    cos(stored, re-embedded) = 1.000000  OK
      negative control  cos(vec[ord 0, doc 4983], text[ord 1861, doc 9967265]) = 0.645611  (must be clearly < 1.0)

==========================================================================
[5] Smoke retrieval — ONE query, wiring check only, NO metrics
==========================================================================
  qid                 : 1
  query               : 0-dimensional biomaterials show inductive properties.
  gold docids (qrels) : ['31715818']
  query instruction   : 'Represent this sentence for searching relevant passages: '
  query vector        : shape=(1, 768) dtype=float32 norm=1.000000

  DENSE top-5 (FAISS IndexFlatIP, score == cosine):
    1. ordinal=2531  docid=14103509 cos=0.5819 resolves=True
       Mechanistic Fracture Criteria For The Failure Of Human Cortical Bone A mecha...
    2. ordinal=3467  docid=21456232 cos=0.5780 resolves=True
       A graphene-based platform for induced pluripotent stem cells culture and dif...
    3. ordinal=828   docid=4346436  cos=0.5712 resolves=True
       Nonlinear Elasticity in Biological Gels Unlike most synthetic materials, bio...
    4. ordinal=3798  docid=23763738 cos=0.5648 resolves=True
       New colorimetric cytotoxicity assay for anticancer-drug screening. We have d...
    5. ordinal=2016  docid=10982689 cos=0.5626 resolves=True
       Nanotoxicology: An Emerging Discipline Evolving from Studies of Ultrafine Pa...

  BM25 top-5 (rank_bm25 BM25Okapi, raw untruncated text):
    1. row=151   docid=825728   bm25=9.4899 resolves=True
       Metastatic colonization requires the repression of the epithelial-mesenchyma...
    2. row=1853  docid=10931595 bm25=8.9452 resolves=True
       Geometry, epistasis, and developmental patterning. Developmental signaling n...
    3. row=4887  docid=43385013 bm25=7.5860 resolves=True
       Epithelial and mesenchymal subpopulations within normal basal breast cell li...
    4. row=2161  docid=13231899 bm25=7.4398 resolves=True
       In situ regulation of DC subsets and T cells mediates tumor regression in mi...
    5. row=2891  docid=18953920 bm25=7.0357 resolves=True
       The Epithelial-Mesenchymal Transition Generates Cells with Properties of Ste...

  overlap between the two top-5 lists: 0/5  (an observation, NOT a metric — Stage 2 owns all measurement)

==========================================================================
[6] Resource recap
==========================================================================
  Stage 1 embed peak  : 1.07 GiB reserved / 0.815 GiB allocated  (ceiling 14.5)
  Stage 1 embed time  : 30.92s   total 55.58s
  this script's peak  : 0.25 GiB reserved

==========================================================================
VERDICT
==========================================================================
  All Stage 1 checks passed. Indexes are persisted, aligned, and reloadable.
  Stage 2 (retrieval + metrics) is unblocked — but do not start it until asked.
