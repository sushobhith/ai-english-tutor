# **AI English Tutor**

## **Description**

The AI English Tutor is a tool designed to help individuals improve their spoken English skills, especially for public speaking. This project was initiated to address the need for affordable and flexible English language practice. The primary goal is to assist users in enhancing their fluency and confidence in spoken English through AI-driven interactions.

## **Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

To use the AI English Tutor, follow these simple installation steps:

```bash

# Clone the repository
git clone https://github.com/sushobhith/ai-english-tutor.git

# Navigate to the project directory
cd ai-english-tutor

```

## Usage

Start practicing spoken English with the AI English Tutor by running the following commands:

```python

pip install -r requirements.txt

```
Copy the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.

```python

python main.py

```

The bot also starts a lightweight version endpoint:

```bash
curl http://localhost:8080/version/v2
```

The endpoint returns:

```json
{
  "name": "ai-english-tutor",
  "version": "v2",
  "status": "ok",
  "environment": "development"
}
```

Set `VERSION_SERVER_HOST` or `VERSION_SERVER_PORT` to change the bind address or port.

Adjust the settings according to your preferences and enjoy practicing at your own pace.

## Features

The AI English Tutor offers several unique features:

1. **Affordability:** Significantly more cost-effective compared to other human-based solutions like Cambly.
2. **Flexibility:** Practice English at your convenience, utilizing small pockets of free time in your schedule.
3. **Confidence Building:** Eliminate hesitation by practicing with an AI tutor, providing a judgment-free environment.

## Contributing

If you'd like to contribute to the AI English Tutor project, follow these steps:

1. Fork the project
2. Create a new branch (**`git checkout -b feature/new-feature`**)
3. Commit your changes (**`git commit -am 'Add new feature'`**)
4. Push to the branch (**`git push origin feature/new-feature`**)
5. Submit a pull request

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) - see the [LICENSE.md](https://github.com/sushobhith/ai-english-tutor/blob/main/LICENSE.md) file for details.
