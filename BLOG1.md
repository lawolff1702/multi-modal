# I Built a Search Engine for a Million Comic Panels

Have you ever wanted to search for a *moment* instead of a title? Not "Daredevil #42 Page #3 Panel #7," but the panel where the hero is silhouetted in a doorway. The one where someone yells "POW." The exact line of dialogue you half-remember but can't place. That's how we actually hold stories in our heads, and it's almost nothing like how we're forced to search for them.

So I built a comic book search engine that lets you do exactly that. It turned into a project I've spent a lot of time playing with, so I want to show it off, what it does, the surprisingly fun things it can do, and where I think the same idea goes next.

It runs over the [COMICS dataset](https://github.com/miyyer/comics), 1,229,664 individual comic panels, each one an image plus whatever dialogue and captions were printed in it, all ingested into a single Pinecone full-text-search index. That one detail is what makes the whole thing tick, and I'll come back to it. The short version is that every panel is searchable in a lot of different ways, and you can freely mix and match them.

## The fun part: what it can actually do

Here's where it stops being a database and starts feeling a little magic.

**Search by what a panel *looks like*, in plain English.** Type *"a detective standing in a dark alley at night"* and you get panels that look like that, even though not one of them was ever tagged "detective" or "alley." The system understands the *picture*, not keywords on it. You can also hand it a panel you like and say "more like this one," and it finds visual cousins across a million images.

**Search by what was *said*.** Looking for the panel with the line *"the secret formula"*? Type it in quotes and you get the literal phrase, not a vague semantic neighborhood of science-y panels. Want every panel where a villain mentions a *laboratory* AND a *formula*? That works too, with real boolean logic.

**Search by sound.** Comics are full of onomatopoeia like POW, BANG, and KRAKOOM. Those are almost impossible to find any other way, but they're just text printed in the art, so they're fully searchable. "Show me dramatic action panels that literally say POW" is a real query.

**And here's the combination that sold me on the whole thing.** Picture a query like *"a panel that looks like a superhero mid-punch, but it must contain the word POW, no exceptions."* That's a visual search and a hard text requirement at the same time. The system finds punch-looking panels by sight, then throws out every single one that doesn't literally say POW before it ranks anything. You get exactly what you asked for, with no "close enough" results sneaking in.

That last one is the thing I keep coming back to, because it's a pattern, not a party trick. *Find me things that look like X but are guaranteed to contain Y.* Once you have that, a lot of search problems get easy.

## How it works (the short version)

Under the hood, every panel is one record carrying three different kinds of signal, and each does a job the others can't.

The first is a visual and semantic vector from CLIP, a model trained on hundreds of millions of image and caption pairs until pictures and the words that describe them live in the same mathematical space. That shared space is the trick that lets a text description retrieve an image, since the two become directly comparable. It's great at vibe and visual similarity, and weak at exact words.

The second is a learned sparse keyword vector over the dialogue, basically a smarter version of classic keyword search that knows which words actually carry meaning. This is what keeps "laboratory" and "formula" as findable, weighted keywords instead of blurring them into a general science-y mush.

The third is full-text search over the printed text. This is the precision layer, handling exact phrases, boolean queries, and the all-important ability to demand that results contain something as a hard filter on any of the other searches.

When a query touches more than one of these, the results get merged with Reciprocal Rank Fusion, a simple and robust way to blend ranked lists without having to make their incompatible scores comparable. The upshot is one search box, with the right signal or combination firing depending on what you typed or dragged in.

The interesting design decision is that all three signals live on the **same record**. There's no separate keyword engine over here and vector store over there that I have to keep in sync and reconcile at query time. One panel, one document, every way of searching it attached to it directly.

## Why this isn't really about comics

Comics are a fun playground, but strip away the capes and the pattern is dead general. *Anything* that is simultaneously an image (or audio, or video) **and** some associated text wants exactly this treatment. A few of the wins feel obvious to me.

- **E-commerce and product catalogs.** "Find me a couch that *looks like* this photo, but it must be listed as **velvet** and under a certain price." Visual similarity for the look, full-text and metadata for the non-negotiable constraints. Shoppers describe vibes while inventory lives in exact attributes, and this bridges them.
- **Media and stock libraries.** Millions of photos, video frames, or audio clips with captions, transcripts, or tags. Search by visual feel, then hard-filter by rights, location, or a spoken phrase in the transcript. Same POW-punch pattern, looks like X and guaranteed to mention Y.
- **Technical and scientific documents.** Figures, diagrams, and charts paired with their captions and surrounding text. "Find diagrams that look like this architecture, but only in papers that mention 'retrieval-augmented.'" Visual plus exact-term filtering is how researchers actually hunt.
- **Real estate and marketplaces.** "Homes that look like this interior, must say 'hardwood floors,' in this ZIP." Listing photos carry the feel while the listing text carries the must-haves.
- **Support and knowledge bases with screenshots.** Match a screenshot a user pastes against your docs visually, but require the result to mention the exact error string. No more semantically-similar-but-wrong articles.
- **Memes, brand, and content moderation.** Find visually-similar images that also contain (or *don't* contain) specific text, a genuinely hard problem that this setup makes almost trivial.

The shape remains largely the same. A fuzzy "looks or feels like this" signal, filtered by a "must contain exactly this" signal. The comic panels are just a great way to conceptualize the workflow.

## How Pinecone's full-text indexes make this easy

What makes this workflow tractable is that a single Pinecone index describes all three signals as fields in one schema. The index holds a dense vector field with its dimension and similarity metric, a sparse vector field, and one or more string fields flagged for full-text search. Flagging a string field for full-text search is what turns it from opaque metadata into a real text index, with language-aware tokenization, stemming, and stop-word handling applied at write time. Nothing is federated across systems, so there's no second store to keep in sync and no IDs to reconcile at query time.

That full-text index does two distinct jobs, scoring and filtering, and both run against the same underlying structure. For scoring you pass a `score_by` directive whose type is either `text` or `query_string`. The `text` mode gives you straight BM25 ranking over a field, term frequency and inverse document frequency with length normalization, which is the right default for ranking panels by how well their dialogue matches a bag of words. The `query_string` mode parses a full Lucene-style query instead, so quoted phrases, `AND`/`OR`/`NOT`, grouping, and field-qualified terms like `ocr_text:formula` all work from one query string.

The part that ties the whole system together is filtering. Pinecone's filter language has the usual comparison and set operators (`$eq`, `$ne`, `$gt`, `$in`, `$nin`, and so on) combined with `$and` and `$or`, but on a full-text field it also exposes text-match operators. There's `$match_phrase` for an exact phrase, `$match_all` for "contains all of these terms," and `$match_any` for "contains any of these terms." The important detail is that a filter is evaluated independently of how you score. So I can run a dense vector query for visual similarity and attach `{"ocr_text": {"$match_all": "POW"}}` as a hard prefilter in the same request. The vector decides the ranking, the full-text match decides who is even eligible, and nothing that fails the match ever reaches the results.

That is the POW-punch query reduced to one API call. There's no candidate-generation step, no second pass to re-filter, and no separate search cluster to operate. And because the sparse embeddings come from Pinecone's own inference endpoint (`pinecone-sparse-english-v0`, with `input_type` set to `passage` at write time and `query` at search time), even the keyword vectors are produced inside the same platform rather than a model I have to host and version myself.

That's the part worth internalizing for your own projects. The hard problem in multi-modal search usually isn't the embeddings, it's making fuzzy and exact search cooperate on the same data without gluing three systems together. Putting every signal on one record, with filters that work across all of them, is what turns a "looks like X but must contain Y" idea into about a dozen lines of code.

I built a comic search engine because it was fun. I keep thinking about it because the pattern underneath it is everywhere.