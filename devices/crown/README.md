# Setup for the Crown EEG headset by Neurosity

1. Install dependencies
    ```bash
    npm install
    ```
2. Move the `@neurosity` package to the `public/` folder to make it available to the browser
    ```bash
    mv node_modules/@neurosity public/
    ```

3. Enter your Neurosity e-mail and password in `public/index.html`

4. Start the app
    ```bash
    npm start
    ```

5. Open the app in your (Chrome) browser at `localhost:3000` and check the Debug Console for status messages

6. Press the `Start` button to connect to the headset
