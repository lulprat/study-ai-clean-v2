# Study AI — clean deployment starter

This is a single Git repository for a Netlify static frontend and a Render FastAPI backend. It uses OpenRouter's `openrouter/free` model router by default. Free model availability, capabilities, and rate limits can vary; this is not unlimited service.

## Expected GitHub root

```text
StudyAI_NewRepo/
├── frontend/
│   ├── index.html
│   └── config.js
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   └── preprocessed/
├── netlify.toml
├── .gitignore
└── README.md
```

The `preprocessed` folder contains the indexed Grade 9 textbook retrieval files. Image creation and Gmail/Google sign-in are removed; attaching an image for homework/image understanding remains available. Chat history and notebook data are stored in each user's browser (not a shared multi-device database).

## 1. Create the new GitHub repository

1. Create an empty repository at https://github.com/new, for example `study-ai-clean-v2`.
2. Do not select the option to add a README, .gitignore, or license, because those already exist here.
3. Extract the ZIP, then open Command Prompt in the extracted `StudyAI_NewRepo` folder.
4. Run the commands listed below, replacing `YOUR-GITHUB-USERNAME` with the GitHub username:

```bat
git init
git branch -M main
git add .
git commit -m "Initial clean Study AI project"
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/study-ai-clean-v2.git
git push -u origin main
```

If Git asks to authenticate, finish the browser sign-in. If `git commit` asks for a name/email, configure your Git name and email with `git config --global user.name "Your Name"` and `git config --global user.email "your-github-email@example.com"` and retry the commit.

## 2. Deploy the backend to Render

1. In Render, select New → Web Service and connect `study-ai-clean-v2`.
2. Configure:
   - Runtime: Python 3
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. In Environment, add:
   - `OPENROUTER_API_KEY` = the API key from an account managed by a person eligible under OpenRouter's terms
   - `OPENROUTER_MODEL` = `openrouter/free`
   - `OPENROUTER_SITE_NAME` = `Study AI`
   - `OPENROUTER_SITE_URL` = your public Netlify site URL (optional)
4. Never add a real key to GitHub, `frontend/config.js`, or this README. Do not add the old `HF_TOKEN` setting; this backend does not call Hugging Face.
5. Deploy and copy the new Render URL, e.g. `https://your-new-service.onrender.com`.

Check `https://YOUR-RENDER-SERVICE.onrender.com/health`. Expected fields include `status: online`, `ai_provider: OpenRouter`, `ai_configured: true`, `rag_index_loaded: true`, and `section_map_loaded: true`. If the index flags are false, check that all files are in `backend/preprocessed/` in GitHub.

## 3. Connect the frontend to the new Render service

In GitHub, open `frontend/config.js`, press Edit, and replace the placeholder with the exact new Render base URL (no `/health` and no trailing slash). Keep quotation marks around the URL because it is a JavaScript string:

```js
window.STUDY_AI_CONFIG = {
  API_BASE: "https://your-new-service.onrender.com"
};
```

Commit the change. Netlify will redeploy automatically.

## 4. Deploy the frontend to Netlify

1. In Netlify, import the same GitHub repository.
2. Keep Base directory blank, Build command blank, and Publish directory blank. The root-level `netlify.toml` sets `publish = "frontend"`. If the UI requires a publish directory, explicitly enter `frontend`.
3. Deploy and open the new Netlify site.

## 5. Verify

1. Test Render `/health`.
2. Open the Netlify site and send a short question.
3. Test Textbook Mode, Find Study Pages, image upload/understanding, quiz, Study Sources, and the hamburger/sidebar on a phone.
4. If chat fails, inspect Render logs. `401` suggests a key/auth problem; `402` means inspect OpenRouter account/provider limits and billing; `429` commonly indicates rate/usage limits or model capacity. Wait and retry rather than creating keys to evade limits.

## Local backend test (optional)

From the `backend` directory, install requirements and use `backend/.env.example` as a reference to create an untracked `.env` with the real key. Do not commit `.env`.

```bat
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8028
```

For local frontend testing, set `frontend/config.js` to `http://127.0.0.1:8028`, serve `frontend/` with a local static web server, and restore the Render URL before pushing the public site.

## Notes

- Render free services can sleep when inactive, so first requests can be delayed.
- OpenRouter free-model access has rate limits and changing model availability; it is not guaranteed unlimited access.
- This project does not include account sign-in. Chat and notebook data remain local to each browser.


## Textbook-only mode and citations

Textbook Mode is strict: it refuses questions when no sufficiently relevant indexed Grade 9 Mathematics passage is found, and does not fall back to general knowledge. Retrieval removes generic question words, applies relevance thresholds, and suppresses citations for app-identity or current real-world queries. General Mode can still answer outside the book; if Textbook Source is enabled, only relevant retrieved passages are attached as optional context.

All chat and notebook-generation requests use the configured OpenRouter endpoint (`OPENROUTER_MODEL`, default `openrouter/free`). The backend local default port for this version is 8028; Render should use `$PORT` in its start command.
