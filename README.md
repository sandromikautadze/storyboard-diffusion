# Storyboard Generation with Diffusion Models

**Authors**: Sandro Mikautadze, Elio Samaha

<video src="assets/demo_mikautadze_samaha.mp4" control></video>


## **Repo Structure**

In alphabetical order

- `additional_experiments/` contains code for the other approaches we adopted, but failed. We leave them for future work.
- `poster/` contains images generated used for the poster session.
- `src/` contains the core modules and functions used in this work.
- `storyboards/` contains the generated storyboards, grouped by movie and generation format.
- `generate_storyboard.ipynb` contains the code to generate the storyboards. **Use that to try our pipeline**.
- `report_mikautadze_samaha.pdf` contains the report

## How to run 

Create an `.env` file adding the API key from [TogetherAI](https://www.together.ai/) (it's free)
```bash
TOGETHER_API_KEY = your_api_aky
``` 

Open the `generate_storyboard.ipynb` file and follow the instructions.
