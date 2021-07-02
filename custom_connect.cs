using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Input;
using System.IO;
using System.Linq;

public class TheWindow : Window
{
    // NETWORK SETTINGS
    int[] u = { 784, 16, 16, 10 };
    // first sliders
    float lr = 0.005f;
    float momentum = 0.5f;
    // second sliders
    int timeInterval = 0; // ms
    int sampleInterval = 5000;
    int state = 2;
    int number = 10;
    int start = 1;
    int end = 60000;
    // extras
    int penRow = 3;
    int penCol = 1; // add sample pen  

    // slider (min, max, freq)           LR,  MoM,  Seed,  Time, Sample, State, Number, Start,   End, Row,  Col, wseed, cseed,  mix, jump
    readonly double[] sliderMin = {   0.000,  0.0,   0.0, 0.000,      1,     0,      0,     1,     1,   1,    1,     0,     0,    0,    2 };
    readonly double[] sliderMax = {    0.03,  1.0, 32768,  10.0,   5000,     2,     10, 60000, 60000,   7,    7, 32768, 32768,    1,    3 };
    readonly double[] sliderFreq = { 0.0001, 0.01,     1, 0.001,      1,     1,      1,     1,     1,   1,    1,     1,     1, 0.01,    1 };

    // names
    readonly string[] textBoxName = {
        "Neural Network", "Learning Rate", "Naive Momentum", "Seed",
        "Time Interval", "Sample Interval", "State" , "Number", "Start", "End",
        "Rows", "Cols",
        "Weight Seed", "Custom Seed", "Custom Mix", "Custom Jump"};
    readonly string[] buttonName = {
        "Train", "Test", "Add", "Edit",
        "Reset", "Abort", "Show", "Back",
        "Create", "Load", "Save" };

    // visual gaps
    readonly int margin = 9, // outer global
        gapMenu = 3, // inner global                     
        gapMenuSum = 0;
    double gapMenuExtra = 20, tile = 30;

    // efficiency
    float inputThreshold = 0.35f; // visual mnist threshold        
    float weightThreshold = 0.06f; // input to hidden or input to output

    // console
    string console = "";
    int consoleMax = 17; // console lines

    readonly double titelHeight = 38;
    readonly double menWidth = 224;
    readonly int sliderHeight = 24;

    // files 
    FileStream image = null, label = null;
    string path = @"C:\goodgame\one\", netBack = "";

    // colors
    readonly SolidColorBrush brBack = new SolidColorBrush(Color.FromRgb(44, 42, 41));
    readonly SolidColorBrush brMain = new SolidColorBrush(Color.FromRgb(0, 0, 0));
    readonly SolidColorBrush brFont = new SolidColorBrush(Color.FromRgb(205, 199, 168));
    readonly SolidColorBrush brFont2 = new SolidColorBrush(Color.FromRgb(9, 6, 0));
    readonly SolidColorBrush brAdd = new SolidColorBrush(Color.FromRgb(31, 30, 27));
    readonly SolidColorBrush brGlobal = new SolidColorBrush(Color.FromRgb(25, 25, 25));
    readonly SolidColorBrush brButton = new SolidColorBrush(Color.FromRgb(160, 151, 145));

    // layout
    Canvas canGlobal = new Canvas(),
                canMenu = new Canvas(), // train/test, add, edit
                    canClass = new Canvas(), // train/test
                    canAdd = new Canvas(), // add
                canConsole = new Canvas(),
                canVisual = new Canvas(), // all 
            canVisualBackground = new Canvas();

    // network core
    float[] neuron, gradient, weight, delta;// ArrayResize(netinput, nns-inputs);
    int[] pseudoIndex, ust, wst;
    bool[] special;
    int input, hidden, output, layer, hiddenOutput, inputHidden, neuronLen, weightLen;
    int target = -1, prediction = -10, correct, batch = 1, mode = -2, iter;

    // helper
    int[] neuronCorrect, neuronAll;  // cache accuracy for each class
    double[] xst, yst; // x and y step for visualisation positioning
    double[] xst2;

    bool[] classActivation;
    float[] weightBackStep, deltaBackStep, currentData, inputStorage;
    bool isVisual = true, isReady = true, isLeftClicked = false, abort = false;
    int addPosLast = -1, backprop = 0, curPrediction, curTarget, layMax;

    // multifunctional function helper
    readonly string init = "init", clear = "clear", full = "full", glorot = "glorot";

    int buttonHeight, slider1Height, slider2Height, classAccHeight, consoleHeight;
    double initHeight, initWidth, cHeight, cWidth, mnistX = 0, heightAuto = 0;
    string[] textBoxStr = { "", "", "", "" };  // need to fill!
    readonly double[] sliderStart = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // init first
    int[] yGo = { 0, 0, 0, 0, 0, 0 }; // start heights init required: button, sliders1, sliders2, class, console (margin y), all            

    // object stuff
    readonly Button[] button = new Button[12];
    readonly TextBox[] textBox = new TextBox[10];
    readonly Slider[] slider = new Slider[15];
    readonly TextBlock[] textBlock1 = new TextBlock[16];

    // support  
    readonly System.Globalization.CultureInfo ci = System.Globalization.CultureInfo.GetCultureInfo("en-us");
    readonly Typeface tf = new Typeface("TimesNewRoman"); // "Arial"

    // custom init
    int weightSeed = 12345, customSeed = 0, customLayer = 3;
    float customMix = 0.5f;

    [STAThread]
    public static void Main() { new Application().Run(new TheWindow()); }

    // CONSTRUCTOR - LOADED - ONINIT
    private TheWindow() // constructor
    {
        InitHyperParameter();

        // init global and menu positioning
        int insidegapMenu = 4;
        // mnist start x
        mnistX = 2 * margin + 2 * gapMenu + menWidth;
        gapMenuSum = margin + gapMenu;

        buttonHeight = 55; // button menu height
        slider1Height = 3 * sliderHeight + insidegapMenu;
        slider2Height = 6 * sliderHeight + insidegapMenu; // consoleHeight = (initHeight) - (buttonHeight + slider1Height + slider2Height + classAccHeight + 7 * gapMenu + titelHeight);
        classAccHeight = 112;
        consoleHeight = consoleMax * 14 + 0;

        // y start: button, slider1, slider2, class, console,
        yGo[0] = gapMenuSum; // button
        yGo[1] = yGo[0] + buttonHeight + gapMenu; // slider1
        yGo[2] = yGo[1] + slider1Height + gapMenu; // slider2
        yGo[3] = yGo[2] + slider2Height + gapMenu; // class acc
        yGo[4] = yGo[3] + classAccHeight + gapMenu; // console
        yGo[5] = yGo[4] + consoleHeight + gapMenu;

        initHeight = yGo[5] + margin;
        initWidth = menWidth + 3 * margin + 2 * gapMenu + 28 * 9 + 600 + 100 + 30;

        // create backgrounds for main, menu, button, slider, console // background fixed
        DrawingContext dc = ContextHelpMod(true, ref canGlobal);
        dc.DrawRectangle(brMain, null, new Rect(margin, margin, menWidth + 2 * gapMenu, initHeight - 2 * margin)); // menu background
        dc.DrawRectangle(brBack, null, new Rect(margin + gapMenu, yGo[0], menWidth, buttonHeight)); // button
        dc.DrawRectangle(brBack, null, new Rect(margin + gapMenu, yGo[1], menWidth, slider1Height)); // slider 1
        dc.DrawRectangle(brBack, null, new Rect(margin + gapMenu, yGo[4], menWidth, consoleHeight)); // console 
        dc.Close();

        // set window   
        Title = " goodgame|custom 2021"; //        
        Content = canGlobal;
        Background = brGlobal; // new SolidColorBrush(Color.FromRgb(50, 50, 50));

        Width = initWidth; // 24 + 10 + 5 + 5 + 30 + 28*9 + 600 + 100 + 30; // WidthG;
        Height = initHeight + titelHeight; // HeightG;                   
        MinHeight = yGo[4] + titelHeight; //10 + 55 + sliderHeight * 9 + 108 + 5 + 60;
        MinWidth = 700;

        MouseMove += Mouse_Move;
        MouseDown += Mouse_Down;
        SizeChanged += Window_SizeChanged;

        // init boxes, prevent slider exception
        for (int i = 0; i < 4; i++) textBox[i] = new TextBox();

        // create menu stuff: buttons, sliders etc...
        ButtonPack(); SliderPack1();

        InitLogic(glorot);
        CustomInit(customSeed, customMix, customLayer);

        InitRunPack();
        NeuralNetworkClassAccuracy(init);
        Array.Resize<float>(ref inputStorage, neuronLen * 10);

        canVisual.IsHitTestVisible = false;
        canVisualBackground.IsHitTestVisible = false;
        canConsole.IsHitTestVisible = false;

        canGlobal.Children.Add(canMenu);
        canMenu.Children.Add(canAdd);
        canGlobal.Children.Add(canVisualBackground);
        canGlobal.Children.Add(canConsole);
        canGlobal.Children.Add(canVisual);

        ConsoleExec( // "goodgame|one 2020" + "\n" + "\n" + 
            "Custom network initialized" + "\n"
            + "Hidden activation = ReLU" + "\n"
            + "Output activation = softmax" + "\n"
            + "Optimizer = auto-batch" + "\n" + "\n"
            + "Glorot-init, " + NeuralNetworkInfo(full) + "\n", clear);

        // continue in Window_SizeChanged()...
    } // TheWindow end
    // DESTRUCTOR - CLOSED - DEINIT
    ~TheWindow() { }


    // EVENT 
    void Window_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (!isVisual || ((Canvas)this.Content).RenderSize.Width < 700) return;

        InitVisual();
        // draw visual custom network area
        DrawingContext dc = ContextHelpMod(false, ref canVisualBackground);
        dc.DrawRectangle(Brushes.Black, null, new Rect(gapMenuSum * 2 + menWidth, margin, cWidth - (3 * margin + 2 * gapMenu + menWidth), cHeight - 2 * margin - 1)); // visual area
        dc.Close();

        CustomNetworkSample(false, target);
    } // Window_SizeChanged end
    void Mouse_Move(object sender, MouseEventArgs e)
    {
        if (mode == 2) AddSampleSelection(e, e.GetPosition(this).X, e.GetPosition(this).Y);
    }
    void Mouse_Down(object sender, MouseButtonEventArgs e)
    {
        // check which mouse click
        isLeftClicked = e.LeftButton == MouseButtonState.Pressed;
        int gpy = (int)(e.GetPosition(this).Y), gpx = (int)(e.GetPosition(this).X);

        if (mode != 1) TargetSelection();

        if (!isReady) return;

        PruningGrowingSelection();

        // run: default, train, test
        if (mode < 2) ClassAccuracySelection();
        // Add
        if (mode == 2) StorageSelection();
        if (mode == 2) AddSampleSelection(e, gpx, gpy);

        void PruningGrowingSelection()
        {
            if (gpx < xst[1] && gpx > xst[layer] + tile) return; // zone check

            for (int i = 0, j = input; i < layer - 1; i++)
            {
                double xstTmp = xst[i + 1];
                for (int k = 0, kEnd = u[i + 1], pos = -1; k < kEnd; k++, j++)
                    if (xstTmp < gpx && xstTmp + tile > gpx && yst[j] < gpy && yst[j] + tile > gpy)
                    {
                        if (e.RightButton == MouseButtonState.Pressed)
                            TailorMadePruning((pos = j), i);
                        else if (e.LeftButton == MouseButtonState.Pressed)
                            TailorMadeGrowing((pos = j + 1), i); // add clicked position + 1

                        // multifunction: console + refresh visual nn
                        ggExec((!isLeftClicked ? "Pruned " : "Growed ") + "node " + pos + ", "
                            + "in " + u[i] + (i == layer - 1 ? "" : " out " + u[i + 2])
                            + ", L " + (i + 1) + " N " + u[i + 1] + "\n", false, -1);

                        if (mode == 3) textBox[0].Text = NetworkToString(); // nn textbox info
                        return;
                    }
            }

            // PRUNING AND GROWING PART
            void TailorMadePruning(int pos, int neuronLayer)
            {

                for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
                    for (int k = 0, ke = u[i + 1]; k < ke; k++, j++)
                        for (int n = t, ne = u[i], m = w + k; n < t + ne; n++, m += ke)
                            if (pseudoIndex[m] == pos && backprop == 0)
                                special[m] = false;

                for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
                    for (int k = 0, ke = u[i + 1]; k < ke; k++, j++)
                        for (int n = t, ne = u[i], m = w + k; n < t + ne; n++, m += ke)
                            if (pseudoIndex[m] > pos)
                                pseudoIndex[m] -= 1;

                // 1. get the pruned ingoing and outgoing weight positions
                bool[] pruned = new bool[weightLen];
                for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
                    for (int k = 0, ke = u[i + 1]; k < ke; k++, j++)
                        for (int n = t, ne = u[i], m = w + k; n < t + ne; n++, m += ke)
                            if (pos == j || pos == n) // if ingoing or outgoing cons of this neuron
                                pruned[m] = true;

                // 2. shift the pruned weights to the end
                for (int w = 0, b = 0; w < weightLen; w++)
                    if (!pruned[w])
                    {
                        pseudoIndex[b] = pseudoIndex[w];
                        special[b] = special[w];
                        weight[b] = weight[w];
                        delta[b++] = delta[w];
                    }

                // 3. delete the neuron on its layer and resize the network
                u[neuronLayer + 1] -= 1;
                InitLogic();
            } // TailorMadePruning end


            void TailorMadeGrowing(int pos, int neuronLayer)
            {
                // 1. save weights
                float[] weightBack = weight.ToArray(), deltaBack = delta.ToArray();
                int[] pseudoIndexBack = pseudoIndex.ToArray();
                bool[] specialBack = special.ToArray();

                // 2. add this neuron to its layer and resize the network
                u[neuronLayer + 1] += 1;
                InitLogic();

                // 3. save postion of ingoing and outgoing weights of this neuron
                bool[] growed = new bool[weightLen];
                for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
                    for (int k = 0, ke = u[i + 1]; k < ke; k++, j++)
                        for (int n = t, ne = u[i], m = w + k; n < t + ne; n++, m += ke)
                            if (pos == j || pos == n) // if ingoing or outgoing cons of that neuron
                                growed[m] = true;

                // 4. restore weights or add the new ones
                Erratic rnd = new Erratic(FastRand());
                for (int i = 0, w = 0, j = 0; i < layer; i++, w += u[i] * u[i - 1])
                {
                    float sd = (float)Math.Sqrt(6.0f / (u[i] + u[i + 1])); //6.0f / (u[i] + u[i + 1])
                    for (int m = w; m < w + u[i] * u[i + 1]; m++)
                        if (!growed[m]) // restore
                        {
                            pseudoIndex[m] = pseudoIndexBack[j];
                            special[m] = specialBack[j];
                            weight[m] = weightBack[j];
                            delta[m] = deltaBack[j++]; // cache for training
                        }
                        else // create glorot weight
                        {
                            float rndNum = rnd.nextFloat(-sd, sd);
                            weight[m] = rndNum < 0 ? 0.2f * rndNum : rndNum;
                            delta[m] = 0;
                        }
                }

                // custom connect
                for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
                    for (int k = 0, ke = u[i + 1]; k < ke; k++, j++)
                        for (int n = t, ne = u[i], m = w + k; n < t + ne; n++, m += ke)
                            if (pseudoIndex[m] > pos)
                                pseudoIndex[m] += 1;
                            else if (pos == pseudoIndex[m])
                                special[m] = false;

                int cl = customLayer - (2);
                for (int i = 0, j = u[0], t = 0, w = 0, back = layer; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
                {
                    int flip = cl;
                    if (cl + (i + 1) > layer - 1) flip = back - 2;
                    back--;

                    for (int k = 0, right = u[i + 1], left = t + u[i]; k < right; k++, j++)
                        for (int n = t, m = w + k; n < left; n++, m += right)
                            if (pos == j || pos == n)
                            {
                                if (i == layer - 1 && pos == n)
                                {
                                    special[m] = false;
                                    pseudoIndex[m] = 0;
                                }
                                else
                                if (FastRand() / 32767.0f >= 1 - customMix)
                                {
                                    int pos1 = ((int)FastRand() % (ust[i + 3 + flip] - ust[i + 2]));
                                    special[m] = true;
                                    pseudoIndex[m] = pos1 + ust[i + 2];
                                }
                                else
                                {
                                    special[m] = false;
                                    pseudoIndex[m] = 0;
                                }
                            }
                }
            } // TailorMadeGrowing end
        }

        // local functions
        void TargetSelection()
        {
            double xstTmp = xst[layer];
            if (gpx >= xstTmp && gpx <= xstTmp + tile && gpy >= yst[inputHidden] && gpy <= yst[neuronLen - 1] + tile)
                for (int j = inputHidden; j < neuronLen; ++j)
                    if (xstTmp < gpx && xstTmp + tile - 1 > gpx && yst[j] < gpy && yst[j] + tile - 1 > gpy)
                        ggExec("", true, (target = j - (inputHidden))); // multifunction: console + train user target + refresh visual nn
        } // TargetSelection end       

        void ClassAccuracySelection()
        {
            double dcx = (gpx - (gapMenuSum + 8)) / (200.0 / output + 1), dcy = (gpy - (yGo[3] + 10)) / 100.0;
            if (dcx < 0.0 || dcx >= output || dcy < 0 || dcy >= 1) return; // if not inside return
            int cx = (int)dcx;
            if (isLeftClicked && !classActivation[cx] || !isLeftClicked && classActivation[cx])
            {
                classActivation[cx] = !classActivation[cx];
                DrawNeuralClass();
                ConsoleExec("Class " + cx + (!classActivation[cx] ? " deaktivated" : " activated") + " for training\n");
            }
        } // ClassSelection end  
        void StorageSelection()
        {
            double cx = (gpx - (gapMenuSum + 2)) / 22.0, cy = (gpy - (yGo[4] - 16)) / 10.0;
            if (cx < 0 || cx >= 10 || cy < 0 || cy >= 1) return;
            if (isLeftClicked)
            {
                for (int i = 0, store = (int)cx * 784; i < 784; i++, store++)
                    neuron[i] = inputStorage[store];
                ConsoleExec("Load Data " + ((int)cx).ToString() + "\n");
            }
            else //if (e.RightButton == MouseButtonState.Pressed)
            {
                for (int i = 0, store = (int)cx * 784; i < 784; i++, store++)
                    inputStorage[store] = neuron[i];
                ConsoleExec("Save Data " + ((int)cx).ToString() + "\n");
            }
            CustomNetworkSample(false, -1); // target by user, no storage
        } // Storage end

    }
    void AddSampleSelection(MouseEventArgs e, double gpx, double gpy)
    {
        // 1. prepare metric
        double dcx = (gpx - (gapMenuSum)) / 8, dcy = (gpy - yGo[2]) / 8;
        int cx = (int)dcx, cy = (int)dcy, addPos = cx + cy * 28;

        // 2. mouse inside check
        if (dcy >= 28 || dcx >= 28 || dcy < 0 || dcx < 0 || addPos == addPosLast) return;

        // 3. check mouse click
        if (e.LeftButton == MouseButtonState.Pressed)
        {
            for (int i = 0; i < penRow; i++) for (int j = 0; j < penCol; j++)
                    if (cx + i < 28 && cy + j < 28) // inside check
                        if (neuron[cx + i + (cy + j) * 28] == 0)
                            neuron[cx + i + (cy + j) * 28] = 1;

            addPosLast = addPos;
            CustomNetworkSample(false, -1);
        }
        else if (e.RightButton == MouseButtonState.Pressed)
            if (neuron[cx + cy * 28] > 0)
            {
                neuron[cx + cy * 28] = 0;
                addPosLast = addPos;
                CustomNetworkSample(false, -1);
            }
    }

    // INIT
    void InitHyperParameter()
    {
        textBoxStr[0] = NetworkToString();
        textBoxStr[1] = lr.ToString();
        textBoxStr[2] = momentum.ToString();
        textBoxStr[3] = weightSeed.ToString();

        sliderStart[0] = Math.Round(lr, 5);
        sliderStart[1] = momentum;
        sliderStart[2] = weightSeed;
        sliderStart[3] = timeInterval / 1000.0;
        sliderStart[4] = sampleInterval;
        sliderStart[5] = state;
        sliderStart[6] = number;
        sliderStart[7] = start;
        sliderStart[8] = end;
        sliderStart[9] = penRow;
        sliderStart[10] = penCol;
    }

    void InitLogic(string def = "")
    {
        neuronLen = weightLen = 0;
        // 1. get neurons and weights of that custom network
        Array.Resize<int>(ref ust, u.Length + 2);
        Array.Resize<int>(ref wst, u.Length + 2);

        layer = u.Length - 1; // layer count

        for (int n = 0; n < layer + 1; n++) ust[n + 1] = neuronLen += u[n];
        for (int n = 1; n < layer + 1; n++) wst[n] = weightLen += u[n - 1] * u[n];

        // 2. set helper    
        input = u[0]; // input neurons
        output = u[layer]; // output neurons
        hidden = neuronLen - (input + output); // hidden neurons
        inputHidden = neuronLen - output; // size of input and hidden neurons
        hiddenOutput = neuronLen - input; // size of 

        // 3. resize arrays
        Array.Resize<float>(ref currentData, input);
        Array.Resize<float>(ref neuron, neuronLen);
        Array.Resize<float>(ref gradient, hiddenOutput);
        Array.Resize<float>(ref weight, weightLen);
        Array.Resize<float>(ref delta, weightLen);

        Array.Resize<int>(ref pseudoIndex, weightLen);
        Array.Resize<bool>(ref special, weightLen);

        InitVisual();

        if (def == "glorot") GlorotInitialization(weightSeed);
    } // optional = glorot


    void InitVisual(int helpLayer = 0, int helpNeuron = 0)
    {

        Array.Resize<double>(ref xst, layer + 1);
        Array.Resize<double>(ref yst, neuronLen);
        Array.Resize<double>(ref xst2, neuronLen);

        layMax = u[1]; // skip input layer
        for (int i = 2; i < layer + 1; i++) // get layer with max neurons
            if (layMax < u[i]) layMax = u[i];

        cHeight = (((Canvas)this.Content).RenderSize.Height);
        cWidth = (((Canvas)this.Content).RenderSize.Width);

        heightAuto = (cHeight - (28 * 9 + titelHeight / 2)) / 2;

        double height = cHeight - (2 * margin + 2 * gapMenuExtra + tile);
        double width = cWidth - (mnistX + 9 * 28 + 70); //(menWidth + 280 + 100 + 30 - 50); // menuW + mnist + accNeu + gapMenu

        int maxNeurons = layMax > 16 ? layMax : 16;  // avoid extended visual
        double heightStep = (height) / (maxNeurons + 0);
        double widthStep = (width / (layer + 0));
        double yStartVis = margin + gapMenuExtra + (heightStep / 2.0);
        double xStartVis = menWidth + (9 * 28);

        // create visual construction
        double xStep = xStartVis, yStep = -5;

        for (int i = 0, j = 0; i < layer + 1; i++, xStep += widthStep)
        {
            if (maxNeurons > 16 && output <= 10 && i == layer) // output
                yStep = ((heightStep = height / 16) * (16 + 1 - output) / 2.0) + margin + gapMenuExtra;
            else // hidden
                yStep = u[i] < maxNeurons ? (heightStep * (maxNeurons - u[i]) / 2.0) + yStartVis : yStartVis;

            xst[i] = xStep;
            for (int k = 0; k < u[i]; k++, j++, yStep += heightStep)
            {
                yst[j] = yStep;
                xst2[j] = xStep;
            }
        } // ConsoleExec("height " + height + " heightStep " + heightStep + "\n", false);
    } // InitVisual end

    void InitRunPack()
    {
        ClearNetwork(ref neuron);
        ClearCanvasMenu(); // clear whatever it was - run/add/edit

        // run stuff
        SliderPack2(); // NeuralNetworkClassAccuracy(init);

        DrawingContext dc = ContextHelpMod(true, ref canMenu);
        dc.DrawRectangle(brBack, null, new Rect(gapMenuSum, yGo[3], menWidth, classAccHeight)); // class accuracy
        dc.DrawText(new FormattedText("Class Accuracy", ci, 0, tf, 9, brFont), new Point(gapMenuSum + 10, yGo[3] + 0));
        for (int i = 0; i < 6; i++) // accuracy lines 0, 20, 40...
            dc.DrawLine(new Pen(brFont, 0.2), new Point(gapMenuSum + 6, yGo[3] + i * 20 + 10), new Point(gapMenuSum + 219, yGo[3] + i * 20 + 10));
        dc.Close();

        canMenu.Children.Add(canClass);
    }

    // CORE RUN
    void NeuralNetworkRun(bool training)
    {
        // set start values and set timer
        iter = correct = batch = 0;
        isReady = abort = false;
        DateTime elapsed = DateTime.Now, desired = DateTime.Now.AddMilliseconds(180);

        if (training) for (int i = 0; i < weightLen; i++) delta[i] *= momentum;
        if (training) NetworkBackup(); // backstep function

        // get training or test files with images and its labels
        LoadMnist(training);

        int idx = 0, all = 0;
        int curStart = training ? start : 1, len = training ? (end - curStart) + 1 : 10000;

        for (int x = 1; x < len + 1; x++)
        {
            // get input data and label for target
            target = NeuralNetworkInputData();

            // if (x == 59916) continue;

            // exeption handling to prevent index error if output neurons are pruned or restricted
            if (target >= output || training && !classActivation[target]) { continue; }

            // run feedforward (train or test)
            CustomFeedForwardSoftmax(training);

            // check prediction
            bool isCorrect = prediction == target;

            // count prediction overall
            correct += isCorrect ? 1 : 0; all++; // true count

            // count prediction each class
            if (isCorrect) neuronCorrect[target]++; neuronAll[target]++; // store class prediction

            if (training && neuron[inputHidden + target] < 0.99)
            {
                CustomBackpropagation(target);
                if (!isCorrect)
                    CustomOptimizer();
            }

            // goodgame GUI 
            if (SampleInterval() && State(isCorrect) && Number(target) || abort) // visual & console 
            {
                if (abort) break; // check for user abort to cache last visual

                ConsoleExec("Iter =  " + (++idx).ToString()
                    + "   pos = " + (x + curStart - 1).ToString()
                    + "   acc = " + (correct * 100.0 / all).ToString("F2") + "%\n");

                DrawNeuralNetwork(target, prediction, true);  // draw custom network and refresh visuals    

                WaitMilliseconds(timeInterval);   // delay for userinteraction

                GetCurrentSample(); // store visual sample
                iter = 0; // reset sample interval
                desired = DateTime.Now.AddMilliseconds(400); // set user interaction timer
            } // networkInfoCheck
            else if (x % 50 == 0)
                CheckUserInteraction(ref desired);
        } // runs end

        // console info after run
        ConsoleExec(
            (training ? "\nTrain " : "\nTest ") + "accuracy = " + (correct * 100.0 / all).ToString("F2") + "%" + "\n"
            + ("Correct = " + correct + "   incorrect = " + (all - correct) + "\n"
            + "Time = " + (((TimeSpan)(DateTime.Now - elapsed)).TotalMilliseconds / 1000.0).ToString("F2")) + "   backprop = " + backprop + "\n"
            + (abort ? ("\nAbort run!\n") : "\n"));

        SetCurrentSample();
        // if(abort) 
        image.Close(); label.Close();
        abort = false; isReady = true; // 

        // local functions
        int NeuralNetworkInputData()
        {
            for (int n = 0; n < 784; ++n)
                neuron[n] = image.ReadByte() / 255.0f;
            return label.ReadByte();
        }
        void LoadMnist(bool isTrain)
        {
            // load file
            image = new FileStream(!training ? path + @"MNIST_Data\t10k-images.idx3-ubyte" : path + @"MNIST_Data\train-images.idx3-ubyte", FileMode.Open);
            label = new FileStream(!training ? path + @"MNIST_Data\t10k-labels.idx1-ubyte" : path + @"MNIST_Data\train-labels.idx1-ubyte", FileMode.Open);

            // get start data
            image.Seek(16 + (training ? (start - 1) : 0) * 784, 0);
            label.Seek(8 + (training ? (start - 1) : 0), 0);
        } // init MNIST dataset end
        void GetCurrentSample()
        {
            for (int i = 0; i < 784; i++) currentData[i] = neuron[i];
            curPrediction = prediction;
            curTarget = target;
        }
        void SetCurrentSample()
        {
            for (int i = 0; i < 784; i++) neuron[i] = currentData[i];
            prediction = curPrediction;
            target = curTarget;
        }
        // - user interactions
        void CheckUserInteraction(ref DateTime dt)
        {
            if (DateTime.Now < dt) return; // ConsoleExec((cnt++) + " isLeftClicked: " + isLeftClicked + "\n", false);
            Application.Current.Dispatcher.Invoke(System.Windows.Threading.DispatcherPriority.Input, new Action(delegate { }));
            dt = DateTime.Now.AddMilliseconds(Mouse.LeftButton == MouseButtonState.Pressed || Mouse.RightButton == MouseButtonState.Pressed ? 150 : 800);
        }
        void WaitMilliseconds(int ms)
        {
            if (ms < 10) return;
            DateTime des = DateTime.Now.AddMilliseconds(ms);
            while (DateTime.Now < des)
                Application.Current.Dispatcher.Invoke(System.Windows.Threading.DispatcherPriority.Input, new Action(delegate { }));
        }
        // - conditions
        bool SampleInterval() { return ++iter >= sampleInterval; }
        bool Number(int target) { return number == 10 || number == target; }
        bool State(bool pred)
        {
            switch (state)
            {
                case 0: return pred; // correct
                case 1: return !pred; // incorrect                   
                default: return true; // all
            }
        }
    } // NeuralNetworkRun end
    void CustomNetworkSample(bool training, int myTarget)
    {
        CustomFeedForwardSoftmax(training); // prediction check and renew

        if (training)
        {
            NetworkBackup();
            int i = 0, len = isLeftClicked ? 1 : 100;
            for (; i < len; i++)
            {
                CustomBackpropagation(myTarget);
                CustomOptimizer();
                CustomFeedForwardSoftmax(training); // renew the network values for this sample
                if (prediction == myTarget) break;
            }
            ConsoleExec("Train target = " + myTarget + (isLeftClicked ? "" : (" " + i.ToString()) + " times") + "   backprop = " + backprop + "\n");
        } // train or test end
        DrawNeuralNetwork(myTarget, prediction, false); // visual nn
    }

    // CUSTOM NET PART
    void CustomInit(int defCustomSeed = 0, float defCustomMix = 0.5f, int defCustomLayer = 2)
    {
        GlorotInitialization(12345);
        //
        FastSrand(defCustomSeed);

        for (int i = 0; i < weightLen; i++) special[i] = false;

        // custom positions
        for (int n = wst[0]; n < wst[layer - 1]; n += 1) //  if(getParity(n % 27)) //
            if (FastRand() / 32767.0f > 1 - defCustomMix)
                special[n] = true; //  != 4, 8, 16

        int cl = defCustomLayer - 2;
        // custom connect
        for (int i = 0, j = u[0], t = 0, w = 0, back = layer; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
        {
            int flip = (cl + (i + 1) > layer - 1) ? back - 2 : cl;
            back--;

            for (int k = 0, right = u[i + 1], left = t + u[i]; k < right; k++, j++)
                for (int n = t, m = w + k; n < left; n++, m += right)
                    if (i != layer - 1)//  if (special[m])
                    {
                        int pos = ((int)FastRand() % (ust[i + 3 + flip] - ust[i + 2]));
                        if (special[m])
                            pseudoIndex[m] = pos + ust[i + 2];
                        else
                            pseudoIndex[m] = -1;
                    }
                    else
                        pseudoIndex[m] = j;
        }
    } // end CustomInit
    void CustomFeedForwardSoftmax(bool isTraining)
    {
        for (int i = input; i < neuronLen; i++) neuron[i] = 0;
        float scale = 0, max = float.MinValue;
        for (int i = 0, j = input, t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
        {
            for (int k = 0, right = u[i + 1], left = t + u[i]; k < right; k++, j++)
            {
                float nn, dot = 0;
                for (int n = t, m = w + k; n < left; n++, m += right)
                    if ((nn = neuron[n]) > 0)
                        if (special[m])
                            neuron[pseudoIndex[m]] += nn * weight[m];
                        else
                            dot += nn * weight[m];
                dot += neuron[j];
                neuron[j] = dot > 0 || i == layer - 1 ? dot : 0;
                if (i == layer - 1 && dot > max) { max = dot; prediction = k; } // grab maxout here           
            }//--- k ends   
        }
        for (int n = neuronLen - output; n < neuronLen; n++)
            scale += neuron[n] = (float)Math.Exp(neuron[n] - max);
        for (int n = neuronLen - output, m = 0; n < neuronLen; m++, n++)
            neuron[n] /= scale;
    }
    void CustomBackpropagation(int target)
    {
        batch++;

        for (int i = 0; i < neuronLen - input; i++) gradient[i] = 0;

        for (int i = layer, j = neuronLen - 1, w = weightLen - 1, wg = w, ds = (neuronLen - output) - 1, gs = (neuronLen - input) - 1;
            i != 0; i--, w -= u[i + 1] * u[i], ds -= u[i], gs -= u[i + 1]) // layer
        {

            for (int k = 0, left = u[i], jj = j; k != left; k++, jj--) // neuron / gradient
            {
                float gra = 0, nj = neuron[jj];
                if (i == layer) // first check if output or hidden, calc delta for both 
                    gra = output - (k + 1) == target ? 1.0f - nj : -nj; // target - out;
                else
                {
                    if (nj > 0)
                    {
                        for (int n = gs + u[i + 1], right = gs; n > right; n--, wg--)
                            if (!special[wg])
                                gra += weight[wg] * gradient[n];
                    }
                    else
                        wg -= u[i + 1];

                    if (i != 1)
                        for (int n = ds, leftLen = ds - u[i - 1], wd = w - k, pos; n > leftLen; wd -= left, n--)
                            if (special[wd]) if (neuron[(pos = pseudoIndex[wd])] != 0) if (neuron[n] > 0)
                                        gradient[n - input] += weight[wd] * gradient[pos - input];
                }
                gradient[jj - input] += gra; // add gradient to array
            }

            for (int k = 0, left = u[i]; k != left; k++, j--) // neuron / gradient
            {
                float gra = gradient[j - input], nj = neuron[j];

                for (int n = ds, leftLen = ds - u[i - 1], wd = w - k; n > leftLen; wd -= left, n--)
                    if (special[wd])
                    {
                        int pos = pseudoIndex[wd];
                        if (neuron[pos] > 0 && neuron[n] > 0)
                            delta[wd] += neuron[n] * gradient[pos - input];
                    }
                    else if (nj > 0 && neuron[n] > 0)
                        delta[wd] += neuron[n] * gra;
            }
        }
        backprop++; // info counter
    }
    void CustomOptimizer()
    {
        for (int i = 0, mStep = 0; i < layer; i++, mStep += u[i] * u[i - 1]) // layer
        {
            float oneUp = (float)Math.Sqrt(2.0f / (u[i + 1] + u[i])) * (neuronLen / layer * 1.0f) / (batch + 1);
            for (int m = mStep, mEnd = mStep + u[i] * u[i + 1]; m < mEnd; m++) // weight (don't need the neuron loop)
            {
                float del = delta[m], s2 = del * del, wn = weight[m];
                if (s2 > oneUp || wn == 0) continue; // check for overwhelming deltas
                weight[m] += del * lr;
                delta[m] = del * momentum;
            }
        }
        batch = 0;
    }
    // WEIGHT PART
    void GlorotInitialization(int seed = 12345)
    {
        ClearNetwork(ref delta); // reset its delta values
        backprop = 0; // reset backprop count

        Erratic rnd = new Erratic(seed);
        for (int i = 0, w = 0; i < layer; i++, w += u[i] * u[i - 1]) // layer
        {
            float sd = (float)Math.Sqrt(6.0f / (u[i] + u[i + 1]));
            for (int m = w; m < w + u[i] * u[i + 1]; m++) // weights
                weight[m] = rnd.nextFloat(-sd, sd);
        }
    }

    // NETWORK VISUAL AREA
    void NeuralNetworkClassAccuracy(string def = "")
    {
        if (def == init)
        {
            Array.Resize<bool>(ref classActivation, output);
            Array.Resize<int>(ref neuronCorrect, output);
            Array.Resize<int>(ref neuronAll, output);

            for (int i = 0; i < output; i++)
            {
                neuronCorrect[i] = 1;
                neuronAll[i] = output;
                classActivation[i] = true;
            }
        }

        DrawNeuralClass();
    } // optional = init
    void DrawNeuralClass()
    {
        DrawingContext dc = ContextHelpMod(false, ref canClass);
        int outLen = output <= 10 ? output : 10;
        for (int i = 0, classTile = (int)(200.0 / outLen), yg = yGo[3]; i < outLen; i++)
        {
            double acc = neuronCorrect[i] * 100.0 / neuronAll[i];
            byte cp = (byte)(i * 5 + 85);
            dc.DrawRectangle(new SolidColorBrush(classActivation[i] ? Color.FromRgb((byte)(200 - (i + 1) * 10), 74, (byte)(i * 10 + 160)) : Color.FromRgb(cp, cp, cp)),
                null, new Rect(gapMenuSum + 8 + i * (classTile + 1), yg + 10 + 100 - acc, classTile, acc));
            dc.DrawText(new FormattedText(acc.ToString("F1"), ci, 0, tf, 8, brFont2), new Point(gapMenuSum + 10 + i * (classTile + 1), yg + 100));
            dc.DrawText(new FormattedText(i.ToString("F0"), ci, 0, tf, 16, brFont2), new Point(gapMenuSum + 14 + i * (classTile + 1), yg + 74));
        }
        dc.Close();
    }// DrawNeuralClass
    void DrawNeuralNetwork(int targetTmp, int predictionTmp, bool isRun) // refresher too
    {
        // control your visual NN with custom prediction and target
        target = targetTmp; prediction = predictionTmp; // control interface

        if (isRun)
            DrawNeuralClass();
        //
        if (isVisual)
        {
            DrawingContext dc = ContextHelpMod(false, ref canVisual);
            CustomNetworkVisual(ref dc);
            dc.Close();
        }
        Application.Current.Dispatcher.Invoke(System.Windows.Threading.DispatcherPriority.Background, new Action(delegate { }));
    }

    void CustomNetworkVisual(ref DrawingContext dc)
    {
        // store max neuron each layer            
        float[] maxNeuronPos = new float[layer]; //, maxNeuronNeg = new float[layer];
        for (int i = 0, j = input, t = 0; i < layer; i++, t += u[i - 1])
        {
            float maxTmpPos = 0;
            for (int k = 0, kEnd = u[i + 1]; k < kEnd; k++, j++)
            { float nj = neuron[j]; if (nj > maxTmpPos) maxTmpPos = nj; }
            maxNeuronPos[i] = maxTmpPos; // maxNeuronNeg[i] = maxTmpNeg;
        }

        mnistX = 2 * margin + 2 * gapMenu + menWidth;
        cHeight = (((Canvas)this.Content).RenderSize.Height);
        heightAuto = (cHeight - (28 * 9 + titelHeight / 2)) / 2;

        // draw weights
        for (int i = 0, j = u[0], t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
        {
            double xStepIn = xst[i] + 15, xStepOut = xst[i + 1] + 15;
            for (int k = 0, right = u[i + 1], leftLen = t + u[i]; k < right; k++, j++)
            {
                for (int n = t, m = w + k; n < leftLen; n++, m += right)
                {
                    double yStepIn = yst[n] + 15, yStepOut = yst[j] + 15;
                    // CornflowerBlue	100 149 237
                    // MediumOrchid     186  85 211
                    if (i == 0) if (neuron[n] > inputThreshold) // input
                        {
                            int pos = pseudoIndex[m] - 0;
                            if (!special[m]) if (neuron[j] > 0)
                                    dc.DrawLine(new Pen(BrF(66, 166, 0), 0.2), new Point((n % 28) * 9 + mnistX, (n / 28) * 9 + heightAuto), new Point(xStepOut, yStepOut));
                            if (special[m])
                                if (neuron[pos] > 0 || pos >= inputHidden)
                                    dc.DrawLine(new Pen(BrF(50 + 45 - i * (63 / (layer - 1)), 55 // - i * (33 / (layer - 1))
                                       , (i * (93 / (layer - 1))) + 141), 0.1), new Point((n % 28) * 9 + mnistX, (n / 28) * 9 + heightAuto), new Point(xst2[pos] + 15, yst[pos] + 15));
                        }
                    if (i != 0) // hidden 
                    {
                        if (special[m])
                        {
                            int pos = pseudoIndex[m] - 0;
                            if (neuron[pos] > 0) if (neuron[n] > 0)
                                    dc.DrawLine(new Pen(BrF(50 + 45 - i * (63 / (layer - 1)), 55 //-  i * (33 / (layer - 1))
                                       , (i * (93 / (layer - 1))) + 141), 0.5), new Point(xStepIn, yStepIn), new Point(xst2[pos] + 15, yst[pos] + 15));
                        }
                        else
                        {
                            if (neuron[j] > 0) if (neuron[n] > 0) dc.DrawLine(new Pen(BrF(66, 166, 0), 0.33), new Point(xStepIn, yStepIn), new Point(xStepOut, yStepOut));
                        }
                    }
                }
            }//--- k ends   
        }

        // draw neurons
        for (int i = 0, j = u[0], t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1])
        {
            float maxNeu = maxNeuronPos[i];
            double xStepIn = xst[i] + 15, xStepOut = xst[i + 1] + 15;
            for (int k = 0, right = u[i + 1]; k < right; k++, j++)
            {
                float nj = neuron[j], cn = (nj / maxNeu);
                double yStep = yst[j] + 14, xstep = xst[i + 1] + 15;
                if (nj > 0 || i == layer - 1) // hidden neuron visual
                {
                    if (i == layer - 1)
                    {
                        dc.DrawText(new FormattedText(k.ToString(), ci, 0, tf, 10
                            , (k == target ? prediction != target ? Brushes.Red : Brushes.Gold : brBack))//(k == target ? brBack: brBack))
                            , new Point(xstep + 10 //- (nj >= 10.0f ? -2 : 1)
                            , yStep - 21)); //gradient[j - input].ToString("F3") //
                    }
                    dc.DrawEllipse(BrF(cn * 118 + 10, cn * 232 + 23, 0), null, new Point(xstep, yStep), 16, 16);
                    dc.DrawText(new FormattedText(nj.ToString("F3"), ci, 0, tf, 10, brFont2), new Point(xstep - 13 + (nj >= 10.0f ? -2 : 1), yStep - 5)); //gradient[j - input].ToString("F3") //
                }
            }//--- k ends 
        }
        // draw mnist input neurons 
        if (mode != 2)
        {
            for (int i = 0, c = 0; i < 28; i++) for (int j = 0; j < 28; j++, c++)
                    if (neuron[c] > inputThreshold) // cut the lows for peformence
                        dc.DrawRectangle(BrF(255 * neuron[c], 0, 0), null, new Rect(mnistX + 9 * j, 9 * i + heightAuto, 8, 8));
        }
        else // add userpinput too, dirty solution
            for (int i = 0, c = 0; i < 28; i++) for (int j = 0; j < 28; j++, c++)
                {
                    float nj = neuron[c];
                    if (nj > inputThreshold) // cut the lows for peformence
                    {
                        dc.DrawRectangle(BrF(255 * neuron[c], 0, 0), null, new Rect(mnistX + 9 * j, 9 * i + heightAuto, 8, 8));
                        dc.DrawRectangle(BrF(nj * 100, nj * 149, nj * 237), null, new Rect(j * 8 + gapMenuSum, i * 8 + yGo[2], 8 - 1, 8 - 1));  //100, 149, 237
                    }
                }
    }

    // CONSOLE 
    void ggExec(string str, bool training, int myTarget, string def = "") // optional = clear
    {
        ConsoleExec(str, def);
        CustomNetworkSample(training, myTarget);
    }
    void ConsoleExec(string str, string def = "")
    {
        if (def == "clear") console = "";

        console += str;
        string backUp = "";
        int numLines = console.Split('\n').Length;
        if (numLines > consoleMax) // cut into console box
        {
            String[] v = console.Split('\n');
            for (int i = numLines - consoleMax - 1; i < numLines; i++)
                backUp += numLines - 1 == i ? v[i] : v[i] + "\n";
            console = backUp;
        }

        DrawingContext drawingContext = ContextHelpMod(false, ref canConsole);
        drawingContext.DrawText(new FormattedText(console, ci, 0, tf, 11, brFont), new Point(gapMenuSum + 7, yGo[4] + 8));
        drawingContext.Close();
    } // optional = clear
    string NeuralNetworkInfo(string def = "") // optional = full
    {
        string str = "custom network\n" + NetworkToString().Replace(",", "-") + "\n"; // custom net
        str += "Neurons = " + neuronLen + "\n" + "Weights = " + weightLen + "\n";

        if (def == full)
        {
            str += "\nCustom seed = " + customSeed + "\n"; // 
            str += "Custom mix = " + customMix + "\n"; // 
            str += "Custom jump = " + customLayer + "\n"; // 
        }
        return str;
    }

    // BUTTON     
    void Run(string str, bool isTrainig)
    {
        if (!isReady) { ConsoleExec("Is running, abort first!\n"); return; }
        InitRunPack();

        NeuralNetworkClassAccuracy(!isTrainig || mode != 0 ? init : "");

        if (mode <= 1)
        {
            mode = isTrainig ? 0 : 1;
            ggExec(str, false, -1, clear); // visual init   
            NeuralNetworkRun(isTrainig); // train or test run
        }
        else // force menu for user interaction
            ggExec("Ready to run (click again for train or test)\n", false, -1, clear);

        mode = isTrainig ? 0 : 1;

    } // train or test helper 
    void Button_Train(object sender, RoutedEventArgs e) { Run("Train " + NeuralNetworkInfo(full) + "\n", true); } // main buttons
    void Button_Test(object sender, RoutedEventArgs e) { Run("Test " + NeuralNetworkInfo(full) + "\n", false); }
    void Button_Add(object sender, RoutedEventArgs e)
    {
        if (!isReady) { ConsoleExec("Is running, abort first!\n"); return; }
        if (mode == 2 || mode == 1) ClearNetwork(ref neuron); // delete sample after second click

        AddHelper();
        mode = 2;
        ggExec("Add sample\n", false, -1, clear);

        void AddHelper()
        {
            // prepare layout
            ClearCanvasMenu();
            canMenu.Children.Add(canAdd); // add submenu  

            // draw background for input neurons
            DrawingContext dc = ContextHelpMod(false, ref canAdd);
            dc.DrawRectangle(Brushes.Black, null, new Rect(gapMenuSum, yGo[2], menWidth, 223));
            for (int i = 0, c = 0; i < 28; i++) for (int j = 0; j < 28; j++, c++)
                    dc.DrawRectangle(brAdd, null, new Rect(gapMenuSum + 8 * j, 8 * i + yGo[2], 8 - 1, 8 - 1));
            dc.Close();

            // draw background and storage
            dc = ContextHelpMod(true, ref canMenu); // drawingContext.DrawText(new FormattedText(console, ci, 0, tf, 11, brFont), new Point(20, 380));
            dc.DrawRectangle(brBack, null, new Rect(gapMenuSum, yGo[4] - (37 + gapMenu), menWidth, 37)); // background
            dc.DrawText(new FormattedText("Sample Storage", ci, 0, tf, 9, brFont), new Point(gapMenuSum + 10, yGo[4] - 25));
            for (int i = 0; i < 10; i++) // storage for 10 samples, fixed
            {
                dc.DrawRectangle(brAdd, null, new Rect(2 + gapMenuSum + i * 22, yGo[4] - 16, 22 - 1, 10));
                dc.DrawText(new FormattedText(i.ToString(), ci, 0, tf, 9, brFont), new Point(10 + gapMenuSum + i * 22, yGo[4] - 16));
            }
            dc.Close();

        }
    }

    void Button_Edit(object sender, RoutedEventArgs e)
    {
        if (!isReady) { ConsoleExec("Is running, abort first!\n"); return; }

        EditHelper();
        mode = 3;
        ggExec("Edit " + NeuralNetworkInfo(full), false, target, clear);

        void EditHelper()
        {
            ClearCanvasMenu();
            // title box background and text titles
            DrawingContext dc = ContextHelpMod(true, ref canMenu);
            dc.DrawRectangle(brBack, null, new Rect(gapMenuSum, yGo[2], menWidth, yGo[4] - (yGo[2] + gapMenu))); // 2. background for edit
                                                                                                                 //  for (int i = 0; i < 4; i++) dc.DrawText(new FormattedText(textBoxName[i], ci, 0, tf, 8, brFont), new Point(gapMenuSum + 15, yGo[2] + 6 + 30 * i));
            dc.Close();
            // edit boxes         
            for (int i = 0; i < 1; i++) // hyperparms
                MyTextBox(gapMenuSum + 15, yGo[2] + 15 + 30 * i, 193, 20, i, textBoxStr[i], ref canMenu);
            textBox[0].Text = NetworkToString(); // custom network
            //  buttons: create
            for (int i = 8; i < 9; i++)
                MyButton(gapMenuSum + 15, yGo[2] + 23 + 30 * (i - 5), 193, 20, i, buttonName[i], ref canMenu); //MyButton
            button[8].Click += new RoutedEventHandler(Button_Create);// 

            // custom slider
            for (int i = 12, j = 0; i < 15; i++, j++)
                MySlider(176, 1, gapMenuSum + 10, (int)(yGo[4] - titelHeight) - 186 + j * 24, i, ref canMenu);

            // slider[11].ValueChanged += Slider_WeightSeed;
            slider[12].ValueChanged += Slider_CustomSeed;
            slider[13].ValueChanged += Slider_CustomMix;
            slider[14].ValueChanged += Slider_CustomLayer;

            // slider[11].Value = weightSeed;
            slider[12].Value = customSeed;
            slider[13].Value = customMix;
            slider[14].Value = customLayer;
            slider[14].Maximum = layer;
        }
    }

    void Slider_WeightSeed(object sender, RoutedPropertyChangedEventArgs<double> e) { weightSeed = (int)SliderHelp(sender, 11); }
    void Slider_CustomSeed(object sender, RoutedPropertyChangedEventArgs<double> e) { customSeed = (int)SliderHelp(sender, 12); }
    void Slider_CustomMix(object sender, RoutedPropertyChangedEventArgs<double> e) { customMix = (float)SliderHelp(sender, 13); }
    void Slider_CustomLayer(object sender, RoutedPropertyChangedEventArgs<double> e) { customLayer = (int)SliderHelp(sender, 14); }

    void Button_Reset(object sender, RoutedEventArgs e)
    {
        GlorotInitialization((weightSeed == 0 ? 12345 : weightSeed));
        NeuralNetworkClassAccuracy(init);
        ggExec("\nReset deltas and bp-count, initialize Glorot" + "\n" + "\n", false, target);
    }
    void Button_Abort(object sender, RoutedEventArgs e)
    {
        if (isReady) { ConsoleExec("Not running!\n"); return; }
        abort = true; // stop run
    }
    void Button_DisableVisual(object sender, RoutedEventArgs e)
    {
        if (mode == 2 || mode > 3) return;

        if (isVisual = !isVisual) // visual show, swap each click (show / hide)
            ResizeWindow(initWidth, initHeight + titelHeight, 700, yGo[4] + titelHeight, true, "Show");
        else // hide and fix window size                
            ResizeWindow(2 * gapMenuSum + menWidth + 14, initHeight + titelHeight, 0, 0, false, "Hide");
        // local function
        void ResizeWindow(double width, double height, double minWidth, double minHeight, bool resize, string str)
        {
            MinWidth = minWidth;
            MinHeight = minHeight;
            Width = width;
            Height = height;

            WindowState = WindowState.Normal;
            ResizeMode = resize ? ResizeMode.CanResize : ResizeMode.NoResize;
            button[2].IsEnabled = resize; // enable add
            button[6].Content = str;

            ConsoleExec(str + " visual\n");
        }
    }

    void Button_Back(object sender, RoutedEventArgs e)
    {
        if (backprop == 0) { ConsoleExec("No backup found!\n", clear); return; }
        if (NetworkToString() != netBack) { ConsoleExec("Wrong network size!\n", clear); return; }

        weight = weightBackStep.ToArray();
        delta = deltaBackStep.ToArray();

        ggExec("Load last training step\n", false, target);
    }
    void Button_Create(object sender, RoutedEventArgs e)
    {
        // set new custom network + hyperparameters
        string userNetwork = textBox[0].Text.Trim().Replace("-", ",").Replace(".", ",").Replace(",0", "1");
        NetworkTransformStringToIntArray(userNetwork);

        customSeed = (int)slider[12].Value;
        customMix = (float)slider[13].Value;
        customLayer = (int)slider[14].Value;

        InitLogic(glorot);

        slider[14].Maximum = layer;

        NeuralNetworkClassAccuracy(init);

        CustomInit(customSeed, customMix, customLayer);

        GlorotInitialization(weightSeed);

        ggExec("Create " + customMix + " " + NeuralNetworkInfo(full) + "Glorot-init" + "\n", false, 0, clear);

    }

    // EVENT SLIDER         
    double SliderHelp(object sender, int id) { return sliderStart[id] = (sender as Slider).Value; }
    void Slider_LearningRate(object sender, RoutedPropertyChangedEventArgs<double> e) { textBox[1].Text = (lr = (float)SliderHelp(sender, 0)).ToString("F4"); }
    void Slider_Momentum(object sender, RoutedPropertyChangedEventArgs<double> e) { textBox[2].Text = (momentum = (float)SliderHelp(sender, 1)).ToString(); }
    void Slider_Seed(object sender, RoutedPropertyChangedEventArgs<double> e) { textBox[3].Text = (weightSeed = (int)SliderHelp(sender, 2)).ToString(); }
    void Slider_Time(object sender, RoutedPropertyChangedEventArgs<double> e) { timeInterval = (int)(SliderHelp(sender, 3) * 1000); }
    void Slider_Samples(object sender, RoutedPropertyChangedEventArgs<double> e) { sampleInterval = (int)SliderHelp(sender, 4); }
    void Slider_State(object sender, RoutedPropertyChangedEventArgs<double> e) { state = (int)SliderHelp(sender, 5); }
    void Slider_Number(object sender, RoutedPropertyChangedEventArgs<double> e) { number = (int)SliderHelp(sender, 6); }
    void Slider_Start(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (slider[8].Value >= slider[7].Value) start = (int)SliderHelp(sender, 7);
        else slider[7].Value = end;
    }
    void Slider_End(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (slider[8].Value >= slider[7].Value) end = (int)SliderHelp(sender, 8);
        else slider[8].Value = start;
    }
    void Slider_Row(object sender, RoutedPropertyChangedEventArgs<double> e) { penRow = (int)SliderHelp(sender, 9); }
    void Slider_Col(object sender, RoutedPropertyChangedEventArgs<double> e) { penCol = (int)SliderHelp(sender, 10); }

    // UTILITY       
    void ClearNetwork(ref float[] arr) { for (int i = 0, len = arr.Length; i < len; i++) arr[i] = 0; }
    void NetworkBackup()
    {
        netBack = NetworkToString();  // exception check
        weightBackStep = weight.ToArray();
        deltaBackStep = delta.ToArray();
    }
    void ClearCanvasMenu()
    {
        canVisual.Children.Clear();
        canMenu.Children.Clear();
    }
    void NetworkTransformStringToIntArray(string str, string val = "")
    {
        char[] chars = str.ToCharArray();

        Array.Resize<int>(ref u, chars.Count(x => x == ',') + 1);  // resize custom network array

        for (int i = 0, m = 0; i < chars.Length; i++) // transform the network
        {
            if (chars[i] != ',') val += chars[i];
            if (chars[i] == ',' || chars.Length - 2 < i)
            {
                u[m++] = Convert.ToInt16(val); // set neurons each layer
                val = ""; // clear value for next layer
            }
        }
    } // core transform
    string NetworkToString() { return string.Join(",", u); }
    DrawingContext ContextHelpMod(bool isInit, ref Canvas cTmp)
    {
        if (!isInit) cTmp.Children.Clear();
        DrawingVisualElement drawingVisual = new DrawingVisualElement();
        cTmp.Children.Add(drawingVisual);
        return drawingVisual.drawingVisual.RenderOpen();
    }
    Brush BrF(float red, float green, float blue)
    {
        Brush frozenBrush = new SolidColorBrush(Color.FromRgb((byte)red, (byte)green, (byte)blue));
        frozenBrush.Freeze();
        return frozenBrush;
    }

    // MENU PACKS
    void ButtonPack()
    {
        for (int i = 0, c = 0; i < 2; i++) for (int j = 0; j < 4; j++)
                MyButton(gapMenuSum + 4 + j * 55, 5 + yGo[0] + 25 * i, 50, 20, c, buttonName[c++], ref canGlobal);

        button[0].Click += Button_Train;
        button[1].Click += Button_Test;
        button[2].Click += Button_Add;
        button[3].Click += Button_Edit;
        button[4].Click += Button_Reset;
        button[5].Click += Button_Abort;
        button[6].Click += Button_DisableVisual;
        button[7].Click += Button_Back;
    }
    void SliderPack1()
    {
        for (int i = 0; i < 3; i++) // first sliders 
            MySlider(182, yGo[1] + 5, gapMenuSum + 9, i * sliderHeight, i, ref canGlobal);

        slider[0].ValueChanged += Slider_LearningRate;
        slider[1].ValueChanged += Slider_Momentum;
        slider[2].ValueChanged += Slider_Seed;

    }// lr, mom, drop
    void SliderPack2()
    {
        DrawingContext dc = ContextHelpMod(true, ref canMenu); // drawingContext.DrawText(new FormattedText(console, ci, 0, tf, 11, brFont), new Point(20, 380));
        dc.DrawRectangle(brBack, null, new Rect(gapMenuSum, yGo[2], menWidth, 6 * sliderHeight + 4));
        dc.Close();

        for (int i = 0; i < 6; i++) // time, sample, state, number
            MySlider(182, yGo[2] + 5, gapMenuSum + 9, i * sliderHeight, i + 3, ref canMenu);

        slider[3].ValueChanged += Slider_Time;
        slider[4].ValueChanged += Slider_Samples;
        slider[5].ValueChanged += Slider_State;
        slider[6].ValueChanged += Slider_Number;
        slider[7].ValueChanged += Slider_Start;
        slider[8].ValueChanged += Slider_End;
    }// time, sample, state, number...

    // OBJECTS
    void MySlider(int width, int height, int X, int Y, int i, ref Canvas refCan)
    {
        // slider name
        MyLabel(X + 1, height + Y - 1, textBoxName[i + 1], 8, brFont, ref refCan, ref textBlock1[i], true, ref slider[i]);

        slider[i] = new Slider();
        slider[i].Minimum = sliderMin[i];
        slider[i].Maximum = sliderMax[i];
        slider[i].TickFrequency = sliderFreq[i];
        slider[i].Value = sliderStart[i];
        slider[i].IsSnapToTickEnabled = true;
        slider[i].Name = "slider" + i.ToString();
        slider[i].Width = width;
        slider[i].IsSelectionRangeEnabled = true;
        refCan.Children.Add(slider[i]);
        Canvas.SetLeft(slider[i], X - 4);
        Canvas.SetTop(slider[i], height + Y);

        // value with binding
        MyLabel(X + width - 3, height + Y + 2, textBoxName[i + 1], 12, brFont, ref refCan, ref textBlock1[i], false, ref slider[i]);
    }
    void MyButton(int X, int Y, int width, int height, int i, string str, ref Canvas canMeth)
    {
        button[i] = new Button();
        button[i].Width = width;
        button[i].Height = height;
        button[i].Content = str;
        button[i].Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0)); //13, 10, 8
        button[i].Background = brButton;
        button[i].BorderBrush = brButton;
        canMeth.Children.Add(button[i]);
        Canvas.SetLeft(button[i], X);
        Canvas.SetTop(button[i], Y);
    }
    void MyTextBox(int x, int y, int width, int height, int i, string str, ref Canvas refCan)
    {
        textBox[i] = new TextBox();
        textBox[i].Width = width;
        textBox[i].Height = height;
        textBox[i].Text = str; //  textBox[i].Background = new SolidColorBrush(Color.FromRgb(158, 158, 155));
        textBox[i].Background = brButton;
        textBox[i].BorderBrush = brButton;
        textBox[i].Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
        refCan.Children.Add(textBox[i]);
        Canvas.SetLeft(textBox[i], x);
        Canvas.SetTop(textBox[i], y);
    }
    void MyLabel(int x, int y, string text, double font_size, SolidColorBrush br, ref Canvas refCan, ref TextBlock label, bool def, ref Slider sli)
    {
        label = new TextBlock();
        if (def)
            label.Text = text;
        else if (sli.Name == "slider0")
            label.SetBinding(TextBlock.TextProperty, new System.Windows.Data.Binding("Value") { Source = sli, StringFormat = "N4" });
        else if (sli.Name == "slider1" || sli.Name == "slider3" || sli.Name == "slider13")
            label.SetBinding(TextBlock.TextProperty, new System.Windows.Data.Binding("Value") { Source = sli, StringFormat = "N2" });
        else
            label.SetBinding(TextBlock.TextProperty, new System.Windows.Data.Binding("Value") { Source = sli });
        label.FontSize = font_size;
        label.FontFamily = new FontFamily("TimesNewRoman");
        label.Foreground = br;
        refCan.Children.Add(label);
        Canvas.SetLeft(label, x);
        Canvas.SetTop(label, y);
    }

    // FAST RAND // https://software.intel.com/content/www/us/en/develop/articles/fast-random-number-generator-on-the-intel-pentiumr-4-processor.html
    static int g_seed;
    void FastSrand(int seed) { g_seed = seed; }
    int FastRand() { return ((g_seed = (214013 * g_seed + 2531011)) >> 16) & 0x7FFF; } // [0, 32768)

} // TheWindow end

public class DrawingVisualElement : FrameworkElement
{
    private VisualCollection _children;
    public DrawingVisual drawingVisual;
    public DrawingVisualElement()
    {
        _children = new VisualCollection(this);
        drawingVisual = new DrawingVisual();
        _children.Add(drawingVisual);
    }
    protected override int VisualChildrenCount { get { return _children.Count; } }
    protected override Visual GetVisualChild(int index) { return _children[index]; }
} // DrawingVisualElement

// https://jamesmccaffrey.wordpress.com/2019/05/20/a-pseudo-pseudo-random-number-generator/
class Erratic
{
    private float seed;
    public Erratic(float seed2)
    {
        this.seed = this.seed + 0.5f + seed2;  // avoid 0
    }
    public float next()
    {
        double x = Math.Sin(this.seed) * 1000;
        double result = x - Math.Floor(x);  // [0.0,1.0)
        return this.seed = (float)result;
    }
    public float nextFloat(float lo, float hi)
    {
        return (hi - lo) * this.next() + lo;
    }
};