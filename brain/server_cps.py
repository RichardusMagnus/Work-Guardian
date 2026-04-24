import sys
import asyncio
import aiomqtt
import json
import time

# --- FIX PER WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -----------------------

# --- VARIABILI GLOBALI PER IL CERVELLO ---
last_alert_time = 0
COOLDOWN_SECONDS = 10  # Aspetta 10 secondi prima di mandare un nuovo allarme

async def process_logic(client, topic, data):
    global last_alert_time
    
    # 1. Logica Drone
    if "drone" in topic:
        detections = data.get("detections", [])
        
        # Cerca se tra le cose viste c'è una "person" con una confidenza buona (> 60%)
        person_detected = any(d.get("label") == "person" and d.get("conf", 0.0) > 0.6 for d in detections)

        if person_detected:
            current_time = time.time()
            # Controlla se è passato il tempo di Cooldown
            if current_time - last_alert_time > COOLDOWN_SECONDS:
                print("\n⚠️ PERICOLO: Rilevata persona nell'area di scavo!")
                print("📡 Invio allarme immediato allo smartwatch...")
                
                # Prepara e spedisce il pacchetto ad Arduino / Smartwatch Virtuale
                alert_payload = {"msg": "PERSONA IN AREA!", "type": "VISUAL"}
                await client.publish("cantiere/allarmi", payload=json.dumps(alert_payload))
                
                # Resetta il timer
                last_alert_time = current_time

    # 2. Logica Orologio (Telemetria)
    if "orologio" in topic and "operaio_1" in topic:
        hr = data.get("heart_rate", 80)
        if hr > 120:
            print(f"🚨 MEDICO: Battito anomalo rilevato ({hr} bpm)")

async def main():
    async with aiomqtt.Client("localhost") as client:
        print("✅ Server CPS Centrale in ascolto...")
        await client.subscribe("cantiere/#") # Ascolta tutto quello che succede nel cantiere
        async for message in client.messages:
            try:
                payload = json.loads(message.payload.decode())
                await process_logic(client, str(message.topic), payload)
            except Exception:
                pass # Ignora i pacchetti malformati

if __name__ == "__main__":
    asyncio.run(main())