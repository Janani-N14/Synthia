from graph import app

# Choose mode: "prompt" or "upload"
MODE = "prompt"  # or "prompt"

if __name__ == "__main__":
    print("🚀 Running SYNTHIA via LangGraph...")
    try:
        if MODE == "prompt":
            final_state = app.invoke({
                "regenerate_count": 0,
                "prompt": "Generate a sample dataset of online grocery shopping apps with price, ratings, delivery_speed",
            })
        elif MODE == "upload":
            final_state = app.invoke({
                "regenerate_count": 0,
                "prompt": "",
                "csv_path": "/content/drive/MyDrive/Synthia/data/googleplaystore.csv"
            })

        print("\n🎉 Pipeline completed!")
        print("📝 Final Quality Report:")
        for key, val in final_state["quality_report"].items():
            print(f" - {key}: {val}")

    except Exception as e:
        print("❌ Error occurred:", str(e))
