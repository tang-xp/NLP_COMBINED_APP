
// This function is called by the main script when the user selects the XP model.
// It checks the selected data and decides if it needs to ask follow-up questions.
function xp_processSelectedData(data) {
    // Check for missing information in a specific order
    if (!data.event_title) {
        askForMissingInfo('event_title', "What is the name of this event?", data);
    } else if (!data.date) {
        askForMissingInfo('date', "I'm missing a date for the event. What day is it?", data);
    } else if (!data.time) {
        askForMissingInfo('time', "And what time is the event?", data);
    } else {
        // If nothing is missing, go to the final confirmation step
        displayFinalConfirmation(data);
    }
}

// This function handles the user's reply WHEN we are waiting for info for the XP model.
// It is responsible for calling the XP-specific normalization endpoint.
async function xp_handleMissingInfoResponse(responseText, missingInfoType, pendingEventData) {
    const lowerResponse = responseText.toLowerCase().trim();

    if (lowerResponse === 'cancel') {
        addMessage("Okay, I've cancelled this request.", 'assistant');
        resetState(); // This is a global helper function from the main script
        return;
    }

    
    // 1. Validate the user's input first.
    if (xp_validateInput(responseText, missingInfoType)) {
        showTypingIndicator();
        
        // 2. If valid, send it to YOUR backend endpoint to be normalized.
        try {
            const response = await fetch('/xp_normalize_text', { // <-- Using YOUR specific route
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: responseText, type: missingInfoType })
            });

            const data = await response.json();
            hideTypingIndicator();
            
            if (data.error) {
                addMessage("I had trouble understanding that format. Could you try rephrasing?", 'assistant');
                return;
            }

            // 3. Update the pending data with the normalized text.
            pendingEventData[missingInfoType] = data.normalized_text;
            
            // 4. Loop back to see if anything else is missing.
            xp_processSelectedData(pendingEventData);

        } catch (error) {
            handleError(error);
        }

    } else {
        // If the input is NOT valid, show an error and ask again.
        let errorMessage = `Sorry, '${responseText}' doesn't seem like a valid ${missingInfoType}. `;
        if (missingInfoType === 'time') {
            errorMessage += "Please try a format like '3pm' or '15:00'.";
        } else if (missingInfoType === 'date') {
            errorMessage += "Please try a format like 'tomorrow', 'Friday', or '11/9'.";
        }
        addMessage(errorMessage, 'assistant');
    }
}

// This is YOUR specific validation logic.
function xp_validateInput(text, type) {
    const lowerText = text.toLowerCase().trim();
    if (!lowerText) return false;

    if (type === 'time') {
        const timeKeywords = ['noon', 'midnight', 'tonight'];
        if (timeKeywords.includes(lowerText)) return true;
        const timeRegex = /^((\d{1,2}(:\d{2})?(\s*(am|pm))?))$/;
        return timeRegex.test(lowerText);
    }

    if (type === 'date') {
        const numericDateRegex = /^\d{1,2}[\/\-\s]\d{1,2}$/;
        if (numericDateRegex.test(lowerText)) return true;
        const hasLetter = /[a-zA-Z]/.test(lowerText);
        if (!hasLetter) return false;
        const dateKeywords = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'today', 'tomorrow', 'yesterday', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
        const hasKeyword = dateKeywords.some(keyword => lowerText.includes(keyword));
        const hasNumber = /\d/.test(lowerText);
        return hasKeyword || hasNumber;
    }

    if (type === 'event_title') {
        if (lowerText.length < 3 || /^\d+$/.test(lowerText)) return false;
        const fluffWords = ['how', 'what', 'when', 'why', 'yes', 'no', 'okay', 'ok'];
        return !fluffWords.includes(lowerText);
    }
    return true;
}