/////////////////////////////////////////////////////////////////////////////////////////////////////////
// File Name - main.js
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This file contains the main logic for the frontend of the application.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

// /////////////////////////////////////////////////////////////
// info icon functionality
function getInfo(infoMessage) {
    let icon = document.getElementById('info-icon');
    let message = document.getElementById('message');
    let container = document.getElementById('info-container');
    let cross = document.getElementById('cross')

    // Toggle container visibility on icon click
    icon.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent event from bubbling up
        if (container.style.display === "none" || !container.style.display) {
            message.innerText = infoMessage;
            container.style.display = "flex";
            cross.style.display = "block"
        } else {
            container.style.display = "none";
        }
    });

    // Close when clicking  on the cross btn
    cross.addEventListener('click', (e) => {
        // if (container.style.display === "block" && e.target !== icon) {
            container.style.display = "none";
            
        // }
    });

    // Close with ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            container.style.display = "none";
        }
    });
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fucntion Name - calculateDateDifference
// Author - Anuraga Sahoo
// Date - 10 Apr 2025
// Description - This function to calculate month and date difference
/////////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to calculate days/months from current date
function calculateDateDifference(inputDate) {
    // Parse the input date (supports formats like "YYYY-MM-DD", "MM/DD/YYYY", etc.)
    const targetDate = new Date(inputDate);
    const currentDate = new Date();
  
    // Check if the input date is valid
    if (isNaN(targetDate.getTime())) {
      return "Invalid date!";
    }
  
    // Calculate time difference in milliseconds
    const timeDifference = currentDate - targetDate;
  
    // Convert milliseconds to days
    const daysDifference = Math.floor(timeDifference / (1000 * 60 * 60 * 24));
  
    // Calculate approximate months (assuming 30 days/month)
    const approxMonths = Math.floor(daysDifference / 30);
  
    // Calculate precise months (accounts for varying month lengths)
    let preciseMonths;
    const currentYear = currentDate.getFullYear();
    const currentMonth = currentDate.getMonth();
    const targetYear = targetDate.getFullYear();
    const targetMonth = targetDate.getMonth();
  
    preciseMonths = (currentYear- targetYear) * 12 + ( currentMonth - targetMonth );
    
    // Adjust if the target day is earlier in the month than the current day
    if (targetDate.getDate() < currentDate.getDate()) {
      preciseMonths--;
    }
    console.log("days = ", daysDifference, "months",approxMonths, "preciseMonths",preciseMonths)
  
    return {
      days: daysDifference,
      approxMonths: approxMonths,
      preciseMonths: preciseMonths,
    };
  }
  


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fucntion Name - formatCurrency
// Author - Anuraga Sahoo
// Date - 10 Apr 2025
// Description - This function is used to format the currency in the form of ₹.
// function for comaseparated inr or format a number into inr
/////////////////////////////////////////////////////////////////////////////////////////////////////////


function formatCurrency(value){
    if (typeof value !== 'number' || value < 0) {
        return 'Invalid value'; // Handle non-numbers/negative values
      }
    
      if (value < 100000) {
        // Format values less than 1 Lac (1,00,000)
        return `₹${value.toLocaleString('en-IN', { 
          maximumFractionDigits: 2,
          minimumFractionDigits: 2 
        })}`;
      } else if (value < 10000000) { // 1,00,000 ≤ value < 1,00,00,000 (1 Cr)
        const inLac = value / 100000;
        return `₹${inLac.toLocaleString('en-IN', { 
          maximumFractionDigits: 2,
          minimumFractionDigits: 2 
        })} Lac`;
      } else { // value ≥ 1,00,00,000 (1 Cr)
        const inCr = value / 10000000;
        return `₹${inCr.toLocaleString('en-IN', { 
          maximumFractionDigits: 2,
          minimumFractionDigits: 2 
        })} Cr`;
      }
    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fucntion Name - formatPercentage
// Author - Ojas Ulhas Dighe
// Date - 28th Mar 2025
// Description - This function is used to format the percentage in the form of 0.00%.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function formatPercentage(value) {
    return `${parseFloat(value).toFixed(2)}%`;
}


// //////////////////////////////////////////////////////////////
// Fucntion Name - getFormattedDateTime
// Author - Anuraga Sahoo
// current date and time format

function getFormattedDateTime() {
    const date = new Date();
    
    // Get date components
    const day = String(date.getDate()).padStart(2, '0');
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = monthNames[date.getMonth()];
    const year = String(date.getFullYear()).slice(-2);
    
    // Get time components
    let hours = date.getHours();
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    
    // Convert to 12-hour format
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
    hours = String(hours).padStart(2, '0');

    return `${day} ${month} ${year}, ${hours}:${minutes} ${ampm}`;
}

// ////////////////////////////////////////////////////////////////////
// Fucntion Name - updateRecomendationColor
// Author - Anuraga Sahoo
// update recomendation colour
function updateRecomendationColor(recomendation){
    const recDiv = document.getElementById('recDiv')
    if(recomendation === "Strong Buy" || recomendation === "Buy"){
        recDiv.style.background="#30a166"
    }
    else if(recomendation === "Hold" ){
        recDiv.style.background="orange"
    }
    else{
        recDiv.style.background="#f43f5e"
    }
}

// //////////////////////////////////////////////////////////////////////////
// Fucntion Name - classifyCompanyINR
// Author - Anuraga Sahoo
// Function to classify a company based on INR market cap
function classifyCompanyINR(marketCapInr) {
    if (marketCapInr >= 80000e7) {          // ₹80,000 crore or more
      return "Large-Cap";
    } else if (marketCapInr >= 5000e7) {    // ₹5,000 crore to ₹80,000 crore
      return "Mid-Cap";
    } else {                                // Less than ₹5,000 crore
      return "Small-Cap";
    }
  }


// ////////////////////////////////////////////////////////////////////////
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    card.classList.toggle('expanded');

    
    const chevron = card.querySelector('.chevron-icon');
    chevron.classList.toggle('chevron-up');
}

// Ensure the fundamentals card is expanded on load
document.addEventListener('DOMContentLoaded', function() {
    const fundamentalsCard = document.getElementById('fundamentals-card');
    fundamentalsCard.classList.add('expanded');
    const chevron = fundamentalsCard.querySelector('.chevron-icon');
    chevron.classList.add('chevron-up');
});


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - updateFundamentalAnalysisSection
// Author - Ojas Ulhas Dighe
// Date - 28th Mar 2025
// Description - This function updates the fundamental analysis section of the UI
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function updateFundamentalAnalysisSection(fundamentalData) {
    // Basic Info Section
    document.getElementById('companyName').textContent = fundamentalData.basic_info.company_name || 'N/A';
    document.getElementById('sector').textContent = fundamentalData.basic_info.sector || 'N/A';
    document.getElementById('industry').textContent = fundamentalData.basic_info.industry || 'N/A';

    // Valuation Metrics
    document.getElementById('marketCap').textContent = formatCurrency(fundamentalData.valuation_metrics.market_cap);
    document.getElementById('peRatio').textContent = fundamentalData.valuation_metrics.pe_ratio.toFixed(2);
    document.getElementById('forwardPE').textContent = fundamentalData.valuation_metrics.forward_pe.toFixed(2);
    document.getElementById('priceToBook').textContent = fundamentalData.valuation_metrics.price_to_book.toFixed(2);
    document.getElementById('dividendYield').textContent = formatPercentage(fundamentalData.valuation_metrics.dividend_yield);

    // Financial Health
    document.getElementById('totalRevenue').textContent = formatCurrency(fundamentalData.financial_health.total_revenue);
    document.getElementById('grossProfit').textContent = formatCurrency(fundamentalData.financial_health.gross_profit);
    document.getElementById('netIncome').textContent = formatCurrency(fundamentalData.financial_health.net_income);
    document.getElementById('totalDebt').textContent = formatCurrency(fundamentalData.financial_health.total_debt);
    document.getElementById('debtToEquity').textContent = fundamentalData.financial_health.debt_to_equity.toFixed(2);
    document.getElementById('returnOnEquity').textContent = formatPercentage(fundamentalData.financial_health.return_on_equity);

    // Growth Metrics
    document.getElementById('revenueGrowth').textContent = formatPercentage(fundamentalData.growth_metrics.revenue_growth);
    document.getElementById('earningsGrowth').textContent = formatPercentage(fundamentalData.growth_metrics.earnings_growth);
    document.getElementById('profitMargins').textContent = formatPercentage(fundamentalData.growth_metrics.profit_margins);

    // Fundamental Recommendation
    // const fundamentalRecommendationBadge = document.getElementById('fundamentalRecommendationBadge');
    // fundamentalRecommendationBadge.textContent = fundamentalData.recommendation;
    // fundamentalRecommendationBadge.className = `recommendation-badge ${getRecommendationClass(fundamentalData.recommendation)}`;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - getRecommendationClass
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to get the recommendation class based on the recommendation.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function getRecommendationClass(recommendation) {
    const classes = {
        'Strong Buy': 'strong-buy',
        'Buy': 'buy',
        'Hold': 'hold',
        'Sell': 'sell',
        'Strong Sell': 'strong-sell'
    };
    return classes[recommendation] || '';
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - updateCompanySummary
// Author -  Anuraga Sahoo
// Description - This function is used to update the company information or summary of companies
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function updateCompanySummary(text) {
    const companySummary = document.getElementById("companySummary");
    const showAndHideButton = document.getElementById('showMoreSummary');
    const maxLength = 100;

    // Initial setup
    if (text.length > maxLength) {
        // Truncate text and add ellipsis
        companySummary.textContent = text.slice(0, maxLength) + '...' ;
        showAndHideButton.textContent = "Show more";
        showAndHideButton.style.display = 'block'; // Show button
    } else {
        companySummary.textContent = text;
        showAndHideButton.style.display = 'none'; // Hide button if text is short
        return; // Exit if no need for toggle
    }

    // Toggle click handler
    let isExpanded = false;
    showAndHideButton.addEventListener('click', () => {
        if (isExpanded) {
            companySummary.textContent = text.slice(0, maxLength) + '...';
            showAndHideButton.textContent = "Show more";
        } else {
            companySummary.textContent = text;
            showAndHideButton.textContent = "Show less";
        }
        isExpanded = !isExpanded;
    });
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// updated by Anuraga Sahoo
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createPriceChart(chartData) {
    const dates = chartData.map(d => d.date);
    const prices = chartData.map(d => d.price);
    const sma20 = chartData.map(d => d.sma20);
    const sma50 = chartData.map(d => d.sma50);

    const traces = [
        {
            name: 'Price',
            x: dates,
            y: prices,
            type: 'scatter',
            line: { color: '#16a34a' }
        },
        {
            name: '20 SMA',
            x: dates,
            y: sma20,
            type: 'scatter',
            line: { color: '#2563eb' }
        },
        {
            name: '50 SMA',
            x: dates,
            y: sma50,
            type: 'scatter',
            line: { color: '#dc2626' }
        }
    ];

    const layout = {
        paper_bgcolor: '#121218',   // Background of the entire chart
        plot_bgcolor: '#121218', 
        font: {
            color: 'white'  // Adjust text color for readability
          },
        title: 'Price Action',
        autosize: true,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (₹)' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('priceChart', traces, layout);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createIndicatorsChart
// Author - Ojas Ulhas Dighe
// updated by Anuraga Sahoo
// Date - 3rd Mar 2025
// Description - This function is used to create the indicators chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createIndicatorsChart(chartData) {
    const dates = chartData.map(d => d.date);
    const rsi = chartData.map(d => d.rsi);
    const macd = chartData.map(d => d.macd);
    const signal = chartData.map(d => d.signal);

    const traces = [
        {
            name: 'RSI',
            x: dates,
            y: rsi,
            type: 'scatter',
            yaxis: 'y1',
            line: { color: '#8b5cf6' }
        },
        {
            name: 'MACD',
            x: dates,
            y: macd,
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#3b82f6' }
        },
        {
            name: 'Signal',
            x: dates,
            y: signal,
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#ef4444' }
        }
    ];

    const layout = {
        title: 'Technical Indicators',
        autosize: true,
        paper_bgcolor: '#121218',   // Background of the entire chart
        plot_bgcolor: '#121218', 
        font: {
            color: 'white'  // Adjust text color for readability
          },
        xaxis: { title: 'Date' },
        yaxis: { 
            title: 'RSI',
            domain: [0.6, 1]
        },
        yaxis2: {
            title: 'MACD',
            domain: [0, 0.4]
        },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 },
        grid: { rows: 2, columns: 1, pattern: 'independent' }
    };

    Plotly.newPlot('indicatorsChart', traces, layout);
}

// Update chart with new data

// function updateChart(data){
    
//     console.log(data)
//     const dates = data.dates.map(d => d);
//     const historical = data.values.map(d => d);
//     const prediction = data.predictions.map(d => d);
//     console.log(dates)

//     const traces = [

//     // {
//     //     name: 'Price',
//     //     x: dates,
//     //     y: historical,
//     //     type: 'scatter',
//     //     line: { color: '#2563eb' }
//     // },
//     {
//         name: 'Prediction',
//         x: dates,
//         y: prediction,
//         type: 'scatter',
//         line: { color: '#16a34a' }
//     },
// ]


//     const layout = {
//         title: 'Price prediction for next 7 days',
//         autosize: true,
//         paper_bgcolor: '#121218',   // Background of the entire chart
//         plot_bgcolor: '#121218', 
//         font: {
//             color: 'white'  // Adjust text color for readability
//           },

//         xaxis: { title: 'Date' },
//         yaxis: { title: 'Price (₹)' },
//         showlegend: true,
//         legend: { orientation: 'h', y: -0.2 }
//     };

//     Plotly.newPlot('futurePredictChart', traces, layout);

//     // select the days
//     let sevenDay = document.getElementById("7d")
//     let fiftinDay = document.getElementById("15d")
//     let thirtyDay = document.getElementById("30d")

//     sevenDay.addEventListener("click", (e)=>{
//         console.log(e.target.value)
//         let date = data.dates
//         for(i=0; i<= 7; i++){
//             console.log("7d",date[i])
//             return date[i]
//         }
//     })

// }

//  this function is used to show the prediction values
function updateChart(data) {
    // console.log(data);
    let chartCard = document.getElementById('chart-card-hide')
    chartCard.style.display='block'

    // Function to render the chart with specified number of days
    function renderChart(days) {
        // Filter prediction data based on the number of days
        const prediction = data.predictions.slice(0, days);
        // Create dates array to match the prediction length
        // const dates = data.dates.slice(0, data.values.length + prediction.length);
        const dates = data.dates.slice(0,  prediction.length);

        

        const traces = [
            {
                name: 'Prediction',
                x: dates,
                y: prediction,                  
                type: 'scatter',
                line: { color: '#16a34a' }
            }
        ];

        const layout = {
            title: `Price prediction for next ${days} days <br><span style='font-size:14px'>last closing price ${data.lastClose.toFixed(2)}</span>`,
            autosize: true,
            paper_bgcolor: '#121218',
            plot_bgcolor: '#121218',
            font: {
                color: 'white'
            },
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price (₹)' },
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 }
        };

        Plotly.newPlot('futurePredictChart', traces, layout);
    }

    // Select radio buttons
    const sevenDay = document.getElementById("7d");
    const fifteenDay = document.getElementById("15d");
    const thirtyDay = document.getElementById("30d");

    // Event listeners for radio buttons
    sevenDay.addEventListener("click", () => {
        renderChart(sevenDay.value);
    });

    fifteenDay.addEventListener("click", () => {
        renderChart(fifteenDay.value);
    });

    thirtyDay.addEventListener("click", () => {
        renderChart(thirtyDay.value);
    });

    // Set default to 7 days
    sevenDay.checked = true; // Ensure 7d is checked by default
    renderChart(sevenDay.value); // Render chart with 7 days initially
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - stock historical price chart
// Author - Anuraga Sahoo
// Date - 3rd Mar 2025
// Description - This function is used to show the graph of the stock price
/////////////////////////////////////////////////////////////////////////////////////////////////////////


function renderHistoricalStockPriceChart(data){

    const chart_data = data.historical_date_price
    console.log("Historical price data ",chart_data)

    const date = data.historical_date_price.map((dates, index)=>{
        return dates.Date
    })

    function updatePastPrice(inputdays){

    
    
    slicedDate = date.slice(date.length-inputdays, date.length)
    const price = data.historical_date_price.map((closes, index)=>{
        return closes.close
    })
    slicedPrice = price.slice(price.length-inputdays, price.length)

    const traces = [
        {
            name: 'StockPrice',
            x: slicedDate,
            y: slicedPrice,                  
            type: 'scatter',
            line: { color: '#16a34a' }
        }
    ];
    const layout = {
        title: `Price Chart`,
        autosize: true,
        paper_bgcolor: '#121218',
        plot_bgcolor: '#121218',
        font: {
                color: 'white'
            },
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (₹)' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }    
    };
    Plotly.newPlot('HistoricalPriceChart', traces, layout);
}
const past7d = document.getElementById('past7d')
const past15d = document.getElementById('past15d')
const past30d = document.getElementById('past30d')
const past6m = document.getElementById('6m')
const past1y = document.getElementById('1y')
const past5y = document.getElementById('5y')



const max = document.getElementById('max')


past7d.addEventListener('click',()=>{ updatePastPrice( past7d.value)
})
past15d.addEventListener('click',()=>{ updatePastPrice( past15d.value)
})
past30d.addEventListener('click',()=>{ updatePastPrice( past30d.value)
})
past6m.addEventListener('click',()=>{ updatePastPrice( past6m.value)
})
past1y.addEventListener('click',()=>{ updatePastPrice( past1y.value)
})
past5y.addEventListener('click',()=>{ updatePastPrice( past5y.value)
})
max.addEventListener('click',()=>{ updatePastPrice(date.length)
})

// Set default to 365 days
max.checked = true; // Ensure 365 is checked by default
updatePastPrice(date.length); // Render chart with 7 days initially

}





/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - analyzeStock
// Author - Ojas Ulhas Dighe
// updated by Anuraga Sahoo
// Date - 3rd Mar 2025
// Description - This function is used to analyze the stock.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

async function analyzeStock() {
    const stockSymbol = document.getElementById('stockSymbol').value;
    const exchange = document.getElementById('exchange').value;
    const startDate = document.getElementById('startDate').value;


    if (!stockSymbol) {
        showError('Please enter a stock symbol');
        return;
    }

    // Show loading state
    document.getElementById('loadingIndicator').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('errorMessage').classList.add('hidden');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: stockSymbol,
                exchange: exchange,
                startDate: startDate
            })
        });

        const result = await response.json();
        console.log(result)
        if (!result.success) {
            throw new Error(result.error);
        }

        if(result.data.predictTopgainer.length===0){
            document.getElementById('topRecomendation').style.display = "none"
            
        }

        
        updateUI(result.data);
        document.getElementById('tickerName').innerText = stockSymbol.toUpperCase()
    } catch (error) {
        showError(error.message || 'An error occurred while analyzing the stock');
    } finally {
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

// prediction of stocks when click on the toogle buttion

// const toggleBtn = document.getElementById('toggleBtn');
// toggleBtn.addEventListener('click',
        
 async function predictStock() {

    // activate the loader or show the loader after click on the buttion
    document.getElementById('predictLoader').style.display = 'block'
    const stockSymbol = document.getElementById('stockSymbol').value;
try {
const response = await fetch('/api/check', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        symbol: stockSymbol,
        success: true,
        message: "button clicked"
    })
}
);

if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
}


const predictData = await response.json();
// deactivate the loader or show the loader after get the data
document.getElementById('predictLoader').style.display = 'none'
console.log('Success:', predictData);
updateChart(predictData.data.prediction)
} catch (error) {
console.error('Fetch Error:', error);
// Revert toggle if request fails
this.classList.toggle('active');
}
}
// );
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - updateUI
// Author - Ojas Ulhas Dighe
// updated by Anuraga Sahoo
// Date - 3rd Mar 2025
// Description - This function is used to update the UI.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function updateUI(data) {
    // Show results container
    document.getElementById('results').classList.remove('hidden');
    // show charts
    document.getElementById('charts').style.display = 'grid'
//  UPDATE TIME
    document.querySelector('.date').textContent = getFormattedDateTime();
    // Update the color of the recomendation div
    updateRecomendationColor(data.averageRecomendation)
    // update logo of stocks
    document.getElementById('img').setAttribute('src', data.fundamentalAnalysis.basic_info.logoURL)
    // update marketcap tag
    document.getElementById('marketcap-size').innerText =  classifyCompanyINR(data.fundamentalAnalysis.valuation_metrics.market_cap)
    // update month date tag
    let inputDate = document.getElementById('startDate').value
    if(calculateDateDifference(inputDate).approxMonths > 0){
        document.getElementById('month-date-tag').innerText = `Invested in ${calculateDateDifference(inputDate).preciseMonths} Month ago`
    }
    else{
        document.getElementById('month-date-tag').innerText = `Invested in ${calculateDateDifference(inputDate).days} Days ago`
    }

    // update info icon or add click event on info icon
    getInfo('infoMessage lorenm loem lorem lorem lorem')

    // update or add company info or summary
    // document.getElementById('companySummary').textContent = data.fundamentalAnalysis.basic_info.companyBusinessSummary
    updateCompanySummary(data.fundamentalAnalysis.basic_info.companyBusinessSummary)

    // update the price chart
    renderHistoricalStockPriceChart(data.fundamentalAnalysis)
    

    // Update current price and recommendation
    document.getElementById('currentPrice').textContent = formatCurrency(data.currentPrice);
    const recommendationBadge = document.getElementById('recommendationBadge');
    recommendationBadge.textContent = data.averageRecomendation;
    // recommendationBadge.className = `recommendation-badge ${getRecommendationClass(data.recommendation)}`;

    // Update signals list
    const signalsList = document.getElementById('signalsList');
    signalsList.innerHTML = '';
    data.signals.forEach(signal => {
        const li = document.createElement('li');
        li.textContent = signal;
        signalsList.appendChild(li);
        li.setAttribute('class', "analysis-text")
    });

    // Update risk metrics
    const metrics = data.riskMetrics;
    document.getElementById('sharpeRatio').textContent = metrics.sharpeRatio.toFixed(2);
    document.getElementById('volatility').textContent = metrics.volatility.toFixed(2) + '%';
    document.getElementById('maxDrawdown').textContent = metrics.maxDrawdown.toFixed(2) + '%';
    document.getElementById('beta').textContent = metrics.beta ? metrics.beta.toFixed(2) : '-';
    // document.getElementById('alpha').textContent = metrics.alpha ? metrics.alpha.toFixed(2) : '-';
    // document.getElementById('correlation').textContent = metrics.correlation ? metrics.correlation.toFixed(2) : '-';

        // Added Fundamental Analysis Section
        if (data.fundamentalAnalysis) {
            updateFundamentalAnalysisSection(data.fundamentalAnalysis);
            document.getElementById('fundamentalAnalysisSection').classList.remove('hidden');
        } else {
            document.getElementById('fundamentalAnalysisSection').classList.add('hidden');
        }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

    createPriceChart(data.chartData);
    createIndicatorsChart(data.chartData);

    // update new stock recomendation
    // Function to populate the object

    const div = document.getElementById('otherStockRecomendationList');
    
    // Clear existing list items
    div.innerHTML = '';
    
    // Create and append new list items
    for (const [stock, score] of Object.entries(data.predictTopgainer)) {
        const li = document.createElement('h4');
        const divElement = document.createElement('div');
        divElement.appendChild(li)
        li.textContent = `${stock}`;
        div.appendChild(divElement);
        li.setAttribute('class', 'list-group-item')
        divElement.setAttribute('class', 'list-group-item-div');
    }
    // prediction chart update
    document.getElementById('chart-card-hide').style.display="none"

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - showError
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to show the error message.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    popup()
    // Set default date to one year ago
    const defaultDate = new Date();
    defaultDate.setFullYear(defaultDate.getFullYear() - 1);
    document.getElementById('startDate').value = defaultDate.toISOString().split('T')[0];

    // Add enter key listener for stock symbol input
    document.getElementById('stockSymbol').addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            analyzeStock();
        }
    });

    document.getElementById('toggleBtn').addEventListener
    ('click', ()=>{
        predictStock()
    })
});

// Add window resize handler for charts
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        const results = document.getElementById('results');
        if (!results.classList.contains('hidden')) {
            Plotly.relayout('priceChart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            });
            Plotly.relayout('indicatorsChart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true,
                'yaxis2.autorange': true
            });
        }
    }, 250);

    
});
 
// function Name popup
// Author : Anuraga Sahoo
// update disclamour functionality 
function popup() {
     // Disclaimer functionality
     const disclaimerToggle = document.getElementById('disclaimerToggle');
     const disclaimerPopup = document.getElementById('disclaimerPopup');
     const closeDisclaimer = document.getElementById('closeDisclaimer');
     
     
     let disclaimerTimeout;
 
     

     function showDisclaimer() {
        //  disclaimerPopup.classList.remove('hidden');
         disclaimerPopup.style.display = "flex";

         disclaimerTimeout = setTimeout(() => {
             disclaimerPopup.style.display = "none"
         }, 3000); // 3 seconds
     }
 
     function hideDisclaimer() {
        //  disclaimerPopup.classList.add('hidden');
         disclaimerPopup.style.display = "none";

         clearTimeout(disclaimerTimeout);
     }
 
     disclaimerToggle.addEventListener('click', showDisclaimer);
     closeDisclaimer.addEventListener('click', hideDisclaimer);
 
     // Optional: Close when clicking outside popup
     disclaimerPopup.addEventListener('click', (e) => {
        
         if (e.target === disclaimerPopup) {
             hideDisclaimer();
         }
     });
}