// script.js

// Get references to the HTML elements we need to interact with
const imageUploadInput = document.getElementById('imageUpload');
const recommendButton = document.getElementById('recommendBtn');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');

// Add an event listener to the button - triggers when clicked
recommendButton.addEventListener('click', async () => { // Make the function async to use await
    // 1. Get the file selected by the user
    const file = imageUploadInput.files[0]; // Get the first selected file

    // 2. Basic validation - check if a file was actually selected
    if (!file) {
        statusDiv.textContent = 'Please select a product image first!';
        statusDiv.style.color = 'red';
        statusDiv.style.fontWeight = 'normal'; // Reset font weight to normal
        resultsDiv.innerHTML = ''; // Clear any previous results
        return; // Stop if no file is selected
    }

    // --- Implementation Start ---

    // Clear previous results and set loading status
    statusDiv.textContent = 'Uploading image and finding recommendations...';
    statusDiv.style.color = 'black'; // Reset color
    statusDiv.style.fontWeight = 'bold'; // Make it bold to indicate loading
    resultsDiv.innerHTML = ''; // Clear previous images

    // 3. Create FormData to package the file for sending
    const formData = new FormData();
    // The key 'image' MUST match the key expected by your Flask backend
    // in request.files['image']
    formData.append('file', file); // Use 'file' to match the backend

    try {
        // 4. Use the fetch API to send the file to the backend ('/recommend')
        // Replace with your actual backend URL if different
        const backendUrl = 'http://localhost:5000/recommend';

        const response = await fetch(backendUrl, {
            method: 'POST',
            body: formData, // Send the FormData containing the file
            // Headers are often not strictly needed for FormData with fetch,
            // as the browser sets the 'Content-Type' to 'multipart/form-data' automatically.
            // However, you could add them if needed:
            // headers: {
            //   'Accept': 'application/json' // We expect JSON back
            // }
        });

        // 5. Handle the response (success or error)
        if (!response.ok) {
            // If response status is not 2xx (e.g., 400, 500)
            const errorData = await response.json().catch(() => ({})); // Try to get error details
            console.error('Backend Error:', response.status, errorData);
            throw new Error(`Backend error: ${response.status} ${response.statusText}. ${errorData.error || ''}`);
        }

        // 6. If successful, parse the JSON and display the recommended images
        const data = await response.json(); // Parse the JSON response body

        if (data.recommendations && data.recommendations.length > 0) {
            statusDiv.textContent = 'Recommendations found!';
            statusDiv.style.color = 'green';

            data.recommendations.forEach(imageUrl => {
                const imgElement = document.createElement('img');
                // IMPORTANT: Construct the full URL if the backend returns relative paths
                // Since our backend returns full paths starting with /static/...
                // they should work correctly relative to the server root.
                // If the backend returned just '7196.jpg', we'd need:
                // imgElement.src = `http://localhost:5000/static/images/${imageUrl}`;
                // But since it returns '/static/images/7196.jpg', this works:
                
                imgElement.src = `http://localhost:5000${imageUrl}`;
                // imgElement.src = imageUrl;
                imgElement.alt = "Recommendation"; // Add alt text for accessibility
                imgElement.onerror = () => {
                    // Optional: Handle cases where an image URL might be broken
                    console.warn(`Could not load image: ${imageUrl}`);
                    imgElement.alt = "Recommendation (load failed)";
                    // Optionally replace with a placeholder or hide it
                    // imgElement.style.display = 'none';
                };
                resultsDiv.appendChild(imgElement); // Add the image to the results grid
            });

        } else {
            statusDiv.textContent = 'No recommendations found, or an unexpected response format.';
            statusDiv.style.color = 'orange';
        }

    } catch (error) {
        // 7. If error (network error, fetch failure, or error thrown above), display an error message
        console.error('Frontend Fetch Error:', error);
        statusDiv.textContent = `Error fetching recommendations: ${error.message}`;
        statusDiv.style.color = 'red';
    }

    // --- Implementation End ---
});