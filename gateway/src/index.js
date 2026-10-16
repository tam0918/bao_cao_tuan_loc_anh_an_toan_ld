const OPENROUTER_IMAGE_URL = "https://openrouter.ai/api/v1/images";
const SEEDREAM_MODEL = "bytedance-seed/seedream-5-0-flash";
const MAX_CHOICES = 26;

export default {
  async fetch(request, env) {
    if (request.method === "GET") {
      return json({ ok: true, service: "imageflow-gateway" });
    }
    if (request.method !== "POST") {
      return json({ error: { code: "INVALID_REQUEST" } }, 405);
    }

    if (!authorized(request, env.IMAGEFLOW_DEVICE_TOKEN)) {
      return json({ error: { code: "AUTH_FAILED" } }, 401);
    }

    const url = new URL(request.url);
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: { code: "INVALID_REQUEST", message: "Body must be JSON." } }, 400);
    }

    try {
      if (url.pathname === "/v1/classify") return await classify(body, env);
      if (url.pathname === "/v1/edit") return await edit(body, env);
      return json({ error: { code: "INVALID_REQUEST" } }, 404);
    } catch (error) {
      console.error("gateway_error", error);
      return json(
        { error: { code: "UPSTREAM_UNAVAILABLE", message: String(error?.message || error) } },
        502
      );
    }
  },
};

function authorized(request, token) {
  if (!token) return false;
  return request.headers.get("Authorization") === `Bearer ${token}`;
}

async function classify(body, env) {
  const image = validateImage(body.image);
  const choices = Array.isArray(body.choices) ? body.choices : [];

  if (choices.length < 1 || choices.length > MAX_CHOICES) {
    return json(
      { error: { code: "INVALID_REQUEST", message: `choices must contain 1-${MAX_CHOICES} items` } },
      400
    );
  }

  const criteria = {};
  for (const choice of choices) {
    if (!choice || typeof choice.id !== "string" || !choice.id) {
      return json({ error: { code: "INVALID_REQUEST", message: "Each choice needs an id." } }, 400);
    }
    criteria[choice.id] =
      typeof choice.description === "string" && choice.description.trim()
        ? choice.description
        : String(choice.label || choice.id);
  }

  const allowedModels = new Set(["clef-flash", "clef", "clef-omni"]);
  const selector = allowedModels.has(body.model) ? body.model : "clef-flash";
  const modelId = {
    "clef-flash": "@cf/cloudflare/clef-flash",
    "clef": "@cf/cloudflare/clef",
    "clef-omni": "@cf/cloudflare/clef-omni",
  }[selector];

  const response = await env.AI.run(modelId, {
    model: selector,
    state: "Classify the attached construction-site image using only the allowed choices.",
    images: [`data:${image.mime};base64,${image.data_b64}`],
    questions: {
      route: {
        type: "choice",
        instructions:
          typeof body.question === "string" && body.question.trim()
            ? body.question
            : "Which choice best matches this image?",
        criteria,
      },
    },
  });

  const answer = response?.answers?.route;
  if (!answer || answer.type !== "choice" || typeof answer.choice !== "string") {
    return json({ error: { code: "INVALID_SCHEMA" } }, 502);
  }

  const probabilities =
    answer.probabilities && typeof answer.probabilities === "object"
      ? answer.probabilities
      : {};

  const topProbability = Number(probabilities[answer.choice] ?? 0);

  return json({
    choice_id: answer.choice,
    confidence: Number.isFinite(topProbability) ? topProbability : 0,
    probabilities,
    model: selector,
  });
}

async function edit(body, env) {
  if (!env.OPENROUTER_API_KEY) {
    return json(
      { error: { code: "INVALID_REQUEST", message: "OPENROUTER_API_KEY is not configured." } },
      500
    );
  }

  const image = validateImage(body.image);
  const instruction =
    typeof body.instruction === "string" && body.instruction.trim()
      ? body.instruction
      : null;

  if (!instruction) {
    return json({ error: { code: "INVALID_REQUEST", message: "instruction is required." } }, 400);
  }

  const upstream = await fetch(OPENROUTER_IMAGE_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: SEEDREAM_MODEL,
      prompt: instruction,
      input_references: [
        {
          type: "image_url",
          image_url: {
            url: `data:${image.mime};base64,${image.data_b64}`,
          },
        },
      ],
    }),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    console.error("openrouter_error", upstream.status, text.slice(0, 1000));

    if (upstream.status === 429) {
      return new Response(
        JSON.stringify({ error: { code: "RATE_LIMITED" } }),
        {
          status: 429,
          headers: {
            "content-type": "application/json",
            ...(upstream.headers.get("Retry-After")
              ? { "Retry-After": upstream.headers.get("Retry-After") }
              : {}),
          },
        }
      );
    }
    if (upstream.status === 401 || upstream.status === 403) {
      return json({ error: { code: "AUTH_FAILED" } }, 401);
    }
    if (upstream.status >= 500) {
      return json({ error: { code: "UPSTREAM_UNAVAILABLE" } }, 502);
    }
    return json({ error: { code: "INVALID_REQUEST" } }, upstream.status);
  }

  const result = await upstream.json();
  const output = result?.data?.[0];
  if (!output?.b64_json) {
    return json({ error: { code: "INVALID_SCHEMA" } }, 502);
  }

  return json({
    image: {
      mime: output.media_type || "image/png",
      data_b64: output.b64_json,
    },
    model: SEEDREAM_MODEL,
    usage: result.usage || undefined,
  });
}

function validateImage(image) {
  if (
    !image ||
    typeof image.mime !== "string" ||
    typeof image.data_b64 !== "string" ||
    !["image/jpeg", "image/png", "image/webp"].includes(image.mime)
  ) {
    throw new Error("INVALID_IMAGE");
  }
  return image;
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}
