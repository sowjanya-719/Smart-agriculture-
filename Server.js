import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import fetch from "node-fetch";
import * as tf from "@tensorflow/tfjs-node";

const app = express();
const PORT = process.env.PORT || 5000;
const OPENWEATHER_KEY = process.env.OPENWEATHER_KEY; // 🔑 Render env variable

app.use(cors());
app.use(bodyParser.json({ limit: "10mb" }));

let model;

// --- Load ML Model at Startup ---
(async () => {
  try {
    model = await tf.loadLayersModel("file://./model/model.json");
    console.log("✅ ML Model Loaded");
  } catch (err) {
    console.error("⚠️ No ML model found. Leaf scanner will not work until you add one.");
  }
})();

// --- 1. Leaf Disease Prediction ---
app.post("/predict-leaf", async (req, res) => {
  const { leafImage } = req.body;
  if (!leafImage) return res.status(400).json({ error: "No image provided" });

  if (!model) {
    return res.json({ leafStatus: "⚠️ ML model not loaded. Add model in /model folder." });
  }

  try {
    const imageBuffer = Buffer.from(leafImage.split(",")[1], "base64");
    const tensor = tf.node
      .decodeImage(imageBuffer, 3)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims(0);

    const prediction = model.predict(tensor);
    const scores = await prediction.data();

    // Example labels - replace with your trained dataset labels
    const labels = ["Healthy", "Fungal Disease", "Bacterial Disease", "Nutrient Deficiency"];
    const maxIdx = scores.indexOf(Math.max(...scores));

    res.json({
      leafStatus: labels[maxIdx],
      confidence: scores[maxIdx].toFixed(2),
    });
  } catch (err) {
    console.error("Prediction error:", err.message);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// --- 2. Soil Condition Analysis ---
app.post("/analyze-soil", (req, res) => {
  const { ph, moisture } = req.body;
  let result = "";

  if (ph < 5.5) {
    result = "Soil is acidic → Add lime";
  } else if (ph > 7.5) {
    result = "Soil is alkaline → Add gypsum";
  } else if (moisture < 30) {
    result = "Soil moisture is low → Irrigation needed";
  } else if (moisture > 80) {
    result = "Soil too wet → Risk of root rot";
  } else {
    result = "Soil is healthy ✅";
  }

  res.json({ soilResult: result });
});

// --- 3. Pest Risk Prediction (Weather API) ---
app.post("/pest-risk", async (req, res) => {
  const { lat, lon } = req.body;
  if (!lat || !lon) return res.status(400).json({ error: "Location not provided" });

  try {
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_KEY}&units=metric`;
    const response = await fetch(url);
    const weather = await response.json();

    if (!weather.main) {
      return res.status(500).json({ error: "Invalid weather data" });
    }

    const temp = weather.main.temp;
    const humidity = weather.main.humidity;

    let risk = "✅ Low pest risk";
    if (humidity > 70 && temp >= 20 && temp <= 30) {
      risk = "⚠️ High fungal pest risk (humid & warm)";
    } else if (temp > 30 && humidity < 40) {
      risk = "⚠️ Medium risk: hot & dry (mites, thrips)";
    } else if (humidity > 80 && temp < 20) {
      risk = "⚠️ Bacterial disease risk (too damp)";
    }

    res.json({ temperature: temp, humidity, pestRisk: risk });
  } catch (err) {
    console.error("Weather API error:", err.message);
    res.status(500).json({ error: "Failed to fetch weather data" });
  }
});

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`🚀 Backend running on port ${PORT}`);
});
