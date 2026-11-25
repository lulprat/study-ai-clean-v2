    try:
        answer, used_model = run_chat(messages, OPENROUTER_MODEL, max_tokens=3200)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Quiz generation failed: {e}")
    raw = answer.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        obj = json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if a < 0 or b <= a:
            raise HTTPException(status_code=502, detail="Quiz model returned invalid JSON.")
        try:
            obj = json.loads(raw[a:b+1])
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Quiz model returned invalid JSON: {e}")
    questions = obj.get("questions") if isinstance(obj, dict) else None
    if not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=502, detail="Quiz model returned no questions.")
    cleaned_questions = []
    for q in questions[:count]:
        if not isinstance(q, dict):
            continue
        opts = q.get("options") if isinstance(q.get("options"), list) else []
        opts = [clean_text(x) for x in opts[:4]]
        if len(opts) != 4:
            continue
        try:
            ans = int(q.get("answer", 0))
        except Exception:
            ans = 0
        ans = max(0, min(ans, 3))
        cleaned_questions.append({
            "question": clean_text(q.get("question", "")),
            "options": opts,
            "answer": ans,
            "explanation": clean_text(q.get("explanation", "")),
        })
    if not cleaned_questions:
        raise HTTPException(status_code=502, detail="Quiz model returned unusable questions.")
    return {"success": True, "questions": cleaned_questions, "sources": sources, "model": used_model}

@app.get("/notebook/textbook-outline")
def notebook_textbook_outline():
    branches = []
    if isinstance(SECTION_MAP, dict):
        for key, value in SECTION_MAP.items():
            pages = pages_from_data(value)
            if pages:
                branches.append({"title": str(key), "pages": pages})
    if not branches:
        branches = [{"title": f"Chapter {n}: {c["title"]}", "pages": list(range(c["start"], c["end"] + 1))} for n, c in CHAPTERS.items()]
    return {"central_topic": "Grade 9 Mathematics (Ganita Manjari)", "branches": branches}

@app.post("/notebook/task")
def notebook_task(request: NotebookTaskRequest):
    require_openrouter()
    task = request.task.lower().strip()
    max_tokens = 3400 if task in {"quiz", "flashcards", "highlight", "slides", "video", "data-table", "mindmap"} else 2200
    system = (
        "You are Study AI's notebook/source engine. Ground the result in the supplied sources. "
        "Do not silently add unsupported facts. Refer to source names, chapters, and section headings, but do not output book or PDF page numbers. "
        "Exact page locations are available in Find Study Pages. For JSON tasks, output valid JSON only."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": request.instruction + "\n\nSOURCE MATERIAL:\n" + request.source[:120000]},
    ]
    try:
        answer, used_model = run_chat(messages, OPENROUTER_MODEL, max_tokens=max_tokens)
        if task == "audio":
            # Generate script only; a separate TTS call is available on demand.
            pass
        return {"answer": answer, "model": used_model}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Notebook task failed: {e}")

@app.post("/notebook/textbook-highlight")
def textbook_highlight(request: TextbookHighlightRequest):
    require_openrouter()
    context, sources = source_context_for_textbook_portion(request.portion)
    if not context:
        raise HTTPException(status_code=404, detail="No indexed textbook content was found for that chapter or section.")
    instruction = (
        'Return JSON {"lesson":"...","key_phrases":["phrase 1", "phrase 2", ...]}. '
        'Write a faithful, student-friendly lesson based only on the textbook passages. '
        'Select 10-20 short exact phrases copied from the passages as key_phrases. '
        'Keep labels such as [Chapter...], [Book page...], and [PDF page...] where useful so page references remain visible.'
    )
    messages = [
        {"role": "system", "content": "You are a source-grounded textbook highlighter. Return valid JSON only."},
        {"role": "user", "content": f"PORTION: {request.portion}\n\n{instruction}\n\nTEXTBOOK PASSAGES:\n{context}"},
    ]
    try:
        answer, used_model = run_chat(messages, OPENROUTER_MODEL, max_tokens=2800)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Highlight generation failed: {e}")
    # Tolerant JSON extraction to survive occasional fenced output.
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.I)
    try:
        obj = json.loads(clean)
    except Exception:
        a, b = clean.find("{"), clean.rfind("}")
        if a < 0 or b <= a:
            raise HTTPException(status_code=502, detail="Highlight model returned invalid JSON.")
        try:
            obj = json.loads(clean[a:b + 1])
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Highlight model returned invalid JSON: {e}")
    return {"lesson": obj.get("lesson", ""), "key_phrases": obj.get("key_phrases", []), "sources": sources, "model": used_model}

@app.post("/notebook/tts")
def notebook_tts(request: TTSRequest):
    text = clean_text(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")
    raise HTTPException(
        status_code=501,
        detail="Server-generated audio is not configured. Use the browser's built-in speech playback for notebook audio overviews.",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
