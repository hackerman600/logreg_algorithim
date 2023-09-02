#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <random>
#include <cmath>
#include <sstream>



Eigen::MatrixXd matrix_creation(std::string path){
    std::ifstream x_train_file(path);
    std::string line;
    std::vector<std::vector<float>> matrix_2; 
    std::vector<float> matrx_2; 
    std::string val = "";
    int itter = 0;
    int com_itter = 0;
        

    while (std::getline(x_train_file, line)) {
        
        if (line[line.length() - 1] == ','){
            line = "," + line;
        } else {
            line = "," + line + ",";
        }
        
        com_itter = 0;
        itter = 0;

        for (char c : line){
            if (c == ','){
                com_itter += 1;
            }

            if (com_itter == 1 && c != ','){
                val += c;
            }

            if (com_itter == 2){
                matrx_2.push_back(std::stod(val));
                com_itter = 1;
                val = "";
            }

            itter += 1;

            if (itter == line.length()){
                matrix_2.push_back(matrx_2);
                matrx_2.clear();
            }
        }
                
    }

    //CONVERT INTO A MATRIX
    int rows = matrix_2.size();
    int cols = matrix_2[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            matrix(r,c) = matrix_2[r][c];
        }
    }

    return matrix; 

}


void view_image(Eigen::MatrixXd ex, int ind){
    Eigen::MatrixXd imagee = Eigen::Map<Eigen::MatrixXd>(ex.data(), 28, 28);
    sf::RenderWindow window(sf::VideoMode(imagee.rows(),imagee.cols()), "render training example " + std::to_string(ind));

    // Create an SFML image based on the Eigen matrix
    sf::Image image;
    image.create(imagee.cols(), imagee.rows());
    for (int y = 0; y < imagee.rows(); ++y) {
        for (int x = 0; x < imagee.cols(); ++x) {
            // Convert Eigen matrix value to grayscale color
            int value = static_cast<int>(imagee(y, x) * 255);
            sf::Color color(value, value, value);
            image.setPixel(x, y, color);
        }
    }

    // Create an SFML texture and sprite from the image
    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
   
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }




}


std::vector<double> uniform_distribution(double upper, double lower, int sizee){
    std::random_device rd;
    std::mt19937 generate(rd());
    std::uniform_real_distribution<double> val(lower, upper);
    std::vector<double> distribution_values;
    int size = sizee;
    double variance = 0.2;
    double catchr;
    double mean; 

    for (int s = 0; s < size; s++){
            double entry = val(generate);
            double entry2 = -entry;
            distribution_values.insert(distribution_values.begin(), {entry, entry2});
          
    }

    //calculate variance
    for (auto c : distribution_values){
        catchr += c;
    } mean = catchr / distribution_values.size();
    catchr = 0.0;

    for (auto c : distribution_values){
        double sd = (c - mean) * (c - mean);
        catchr += sd;
    } double vec_variance = catchr/distribution_values.size();

    for (int i = 0; i < distribution_values.size(); i++){
        distribution_values[i] *= (std::sqrt(variance/vec_variance));
    }

    catchr, vec_variance = 0.0;

    for (auto c : distribution_values){
        double sd = (c - mean) * (c - mean);
        catchr += sd;
    }  vec_variance = catchr/distribution_values.size();

    //std::cout << "vec_variance = " << vec_variance << std::endl;

     return distribution_values;
}


Eigen::MatrixXd sigmoid_output(Eigen::MatrixXd z_output){
    Eigen::MatrixXd out(z_output.rows(),z_output.cols());
    for (int r = 0; r < out.rows(); r++){
       for (int c = 0; c < out.cols(); c++){
           out(r,c) = 1 / (1 + pow(2.7182818284590452, -1 * z_output(r,c)));
       } 
    }

    return out;
}


Eigen::MatrixXd create_labels(Eigen::MatrixXd y_train, int class_no){
    Eigen::MatrixXd y_out(y_train.rows(), y_train.cols());
    for (int r = 0; r < y_train.rows(); r++){
        for (int c = 0; c < y_train.cols(); c++){
            if (y_train(r,c) == class_no){
                y_out(r,c) = 1;
            } else{
                y_out(r,c) = 0;
            }
        }
    }
    return y_out;
}


std::string create_weights_txtfile(std::vector<Eigen::MatrixXd> weights,std::vector<Eigen::MatrixXd> bias){
    std::ostringstream oss;

    for (int w_i = 0; w_i < weights.size(); w_i++){
        oss << "\n\nWeights for model " + std::to_string(w_i + 1) + " are:\n\n";
        for (int r = 0; r < weights[w_i].rows(); r++){
            for (int c = 0; c < weights[w_i].cols(); c++){
                oss << std::to_string(weights[w_i](r,c));
                if (r == weights[w_i].rows() - 1 && c == weights[w_i].cols() - 1){
                    {};
                } else {
                    oss << ",";
                }
            }
        }

        //std::cout << "bias[w_i] is: " << bias[w_i] << std::endl;
        oss << "\n\nbias for model " + std::to_string(w_i + 1) + " is:\n\n" + std::to_string(bias[w_i](0,0));

    }

    return oss.str();
    
}


double calculate_binary_log_cost(Eigen::MatrixXd output, Eigen::MatrixXd class_labels){
    double loss = 0.0;
    for (int r = 0; r < output.rows(); r++){
        for (int c = 0; c < output.cols(); c++){
            double single_loss = (class_labels(r,c) * std::log(output(r,c)) + (1-class_labels(r,c)) * std::log(1-output(r,c)));
            loss += single_loss;
            //std::cout << "single_loss = " << -single_loss << std::endl;
        }
    }

    //std::cout << loss << std::endl;

    return -loss / 60000.00;
}   


Eigen::MatrixXd calculate_dsigmoid(Eigen::MatrixXd z_output){
    Eigen::MatrixXd dsigmoid_output(z_output.rows(),z_output.cols());
    //std::cout << "dsigmoid_output.shape() = " << dsigmoid_output.rows() << "," << dsigmoid_output.cols() << std::endl;
    for (int r = 0; r < z_output.rows(); r++){
        for (int c = 0; c < z_output.cols(); c++){
            double valuedsig = pow(2.718281828459,-1 * z_output(r,c))/pow((1 + pow(2.718281828459,-1 * z_output(r,c))),2);
            //std::cout << "valuedsig = " << valuedsig << std::endl;
            dsigmoid_output(r,c) = valuedsig;
            // 1 / 1 + e^-z = (0*(1 + e^-z) - 1 * (0 + -e^-z)) / (1 + e^-z)^2 = e^z / (1 + e^-z)^2
        }
    }
    return dsigmoid_output;
}


Eigen::MatrixXd calculate_Dbinarylogcost(Eigen::MatrixXd output, Eigen::MatrixXd class_labels){
    Eigen::MatrixXd Dbinarylogcost_output(class_labels.rows(),class_labels.cols());
    //std::cout << "Dbinarylogcost_output.shape() = " << Dbinarylogcost_output.rows() << "," << Dbinarylogcost_output.cols() << std::endl;
    for (int r = 0; r < class_labels.rows(); r++){
        for (int c = 0; c < class_labels.cols(); c++){
            // -1 * (Y - 2Yp + p) / p-p^2
            double value_Dbinarylogcost = ((class_labels(r,c) - 2 * class_labels(r,c) * output(r,c) + output(r,c)) / output(r,c) - pow(output(r,c),2));
            //std::cout << "value_Dbinarylogcost = " << value_Dbinarylogcost << std::endl;
            Dbinarylogcost_output(r,c) = value_Dbinarylogcost;
        }
    }
    return Dbinarylogcost_output;
}


double calculate_accuracy(Eigen::MatrixXd output,  Eigen::MatrixXd class_labels){
    double tally = 0.0;

    for (int r = 0; r < output.rows(); r++){
        for (int c = 0; c < output.cols(); c++){
            if (std::round(output(r,c)) == class_labels(r,c)){
                tally += 1.0;
            }
        }
    }

    //std::cout << "tally = " << tally << std::endl;
    //std::cout << "output.rows() = " << output.rows() << std::endl;
    double outme = tally / output.rows() * 100;
    //double outme = (tally / output.rows()) * 100;
    //std::cout << "tally / output.rows() * 100 = " << outme << std::endl;
    return outme;
}



std::vector<Eigen::MatrixXd> calculate_gradients(Eigen::MatrixXd x_train, Eigen::MatrixXd output, Eigen::MatrixXd z_output, Eigen::MatrixXd class_labels){
    Eigen::MatrixXd out(10,10);
    std::vector<Eigen::MatrixXd> hello;
    Eigen::MatrixXd Dlogcost  = calculate_Dbinarylogcost(output, class_labels);
    //std::cout << "Dlogcost.shape() = (" << Dlogcost.rows() << "," << Dlogcost.cols() << ")" << std::endl;
    //Eigen::MatrixXd Dlogcost  = -(class_labels(r,c) * std::log(output(r,c)) + (1-class_labels(r,c)) * std::log(1-output(r,c)))  -  (cost_function)
        
        // f = -(Y*log(p) + (1-Y)*log(1-p)) = -Y*log(p) - (1-Y)*log(1-p) =  
        // Y(1-p)       P(1-Y)         Y-Yp        P-Yp             Y - 2Yp + p
       //- -------  +   -------   =   -------  +  -------  =     - ------------- -1 * (Y- 2Yp + p) / p-p^2
        // p(1-p)      p(1-p)         p(1-p)       p(1-p)             p(1-p)

    
    Eigen::MatrixXd Dsigmoid = calculate_dsigmoid(z_output);
    //std::cout << "Dsigmoid.shape() = (" << Dsigmoid.rows() << "," << Dsigmoid.cols() << ")" << std::endl;
    //Eigen::MatrixXd Dsigmoid = 1/(1+e^-z) = 0*(1+e^-z) - 1*-e^-z / (1+e^-z)^2 = (e^-z/(1+e^-z)^2) - e = 2.71828182846 
    Eigen::MatrixXd DW = x_train;  
    //std::cout << "x_train.shape() = (" << x_train.rows() << "," << x_train.cols() << ")" << std::endl;
    Eigen::MatrixXd final_dW = DW.transpose() * Dlogcost.cwiseProduct(Dsigmoid);
    //std::cout << "final_dW.shape() = (" << final_dW.rows() << "," << final_dW.cols() << ")" << std::endl;
    double final_dB = (Dsigmoid.sum() / Dsigmoid.rows()) * (Dlogcost.sum() / Dlogcost.rows());
    hello.push_back(final_dW);
    hello.push_back(Eigen::MatrixXd::Constant(Dsigmoid.rows(), Dsigmoid.cols(),final_dB));

    return hello;
}


int main(){

    int classes = 10;
    int epoch = 1000;
    double total_log_cost;
    double total_log_accuracy; 
    double total_log_accuracy_test; 
    float old_total_log_accuracy;
    double alpha = 0.000001;
   
    //load the data
    Eigen::MatrixXd x_train = matrix_creation("/home/kali/Desktop/machine_learning/neural_networks/from_scratch/cpp_networks/x_train.csv");
    Eigen::MatrixXd x_test = matrix_creation("/home/kali/Desktop/machine_learning/neural_networks/from_scratch/cpp_networks/x_test.csv");
    Eigen::MatrixXd y_train = matrix_creation("/home/kali/Desktop/machine_learning/neural_networks/from_scratch/cpp_networks/y_train.csv");
    Eigen::MatrixXd y_test = matrix_creation("/home/kali/Desktop/machine_learning/neural_networks/from_scratch/cpp_networks/y_test.csv");
     
    std::cout << "x_train.size() = (" << x_train.rows() << "," << x_train.cols() << ")" << std::endl;
    std::cout << "y_train.size() = (" << y_train.rows() << "," << y_train.cols() << ")" << std::endl;
    std::cout << "x_test.size() = (" << x_test.rows() << "," << x_test.cols() << ")" << std::endl;
    std::cout << "y_test.size() = (" << y_test.rows() << "," << y_test.cols() << ")" << std::endl;
    
    //normalise data
    x_train /= 255;

    //View an image.
    view_image(x_train.row(2),2);

    //set the parameters and bias
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::MatrixXd> bias;

    for (int i = 0; i < classes; i++){
        std::vector<double> distribution = uniform_distribution(-0.03,0.03,784);
        Eigen::MatrixXd mymate = Eigen::Map<Eigen::MatrixXd>(distribution.data(), 784,1);
        weights.push_back(mymate);
        bias.push_back(Eigen::MatrixXd::Constant(x_train.rows(), 1, distribution[0]));
    }

    std::cout << "weights[0].size() = (" << weights[0].rows() << "," << weights[0].cols() << ")" << std::endl;
    std::cout << "bias[0].size() = (" << bias[0].rows() << "," << bias[0].cols() << ")" << std::endl;

    //train the algorithim.
    for (int epoch_no = 0; epoch_no < epoch; epoch_no++){
        
        if (epoch_no > 0){
            //std::cout << "itter: " << epoch_no << " Binary log cost: " << total_log_cost/10.0 << std::endl;
            std::cout << "itter: " << epoch_no << " x_train accuracy: " << total_log_accuracy/10.0 << " x_test accuracy: " << total_log_accuracy_test/10.0 << std::endl;

        } total_log_accuracy_test = 0.0; old_total_log_accuracy = total_log_accuracy; total_log_cost = 0.0; total_log_accuracy = 0.0;

        for (int model_no = 0; model_no < classes; model_no++){
            Eigen::MatrixXd z_output = (x_train * weights[model_no] + bias[model_no]).rowwise().sum();
            //std::cout << "z_output.shape() = (" << z_output.rows() << "," << z_output.cols() << ")" << std::endl;
            Eigen::MatrixXd output = sigmoid_output(z_output);//std::cout << "z_output = (" << z_output.rows() << "," << z_output.cols() << ")" << std::endl;  
            //std::cout << "output.shape() = (" << output.rows() << "," << output.cols() << ")" << std::endl;
            Eigen::MatrixXd class_labels = create_labels(y_train, model_no);
            //double log_cost = calculate_binary_log_cost(output, class_labels);
            //total_log_cost += log_cost; 
            double log_accuracy = calculate_accuracy(output, class_labels);
            total_log_accuracy += log_accuracy;
            std::vector<Eigen::MatrixXd> gradients = calculate_gradients(x_train, output, z_output, class_labels);
            weights[model_no] -= alpha * gradients[0];
            bias[model_no] -= alpha * gradients[1];

            //calculate test accuracy
            Eigen::MatrixXd z_output_test = (x_test * weights[model_no] + Eigen::MatrixXd::Constant(x_test.rows(), 1, bias[model_no](0,0))).rowwise().sum();
            //std::cout << "z_output.shape() = (" << z_output.rows() << "," << z_output.cols() << ")" << std::endl;
            Eigen::MatrixXd output_test = sigmoid_output(z_output_test);//std::cout << "z_output = (" << z_output.rows() << "," << z_output.cols() << ")" << std::endl;  
            //std::cout << "output.shape() = (" << output.rows() << "," << output.cols() << ")" << std::endl;
            Eigen::MatrixXd class_labels_test = create_labels(y_test, model_no);
            //double log_cost = calculate_binary_log_cost(output, class_labels);
            //total_log_cost += log_cost; 
            double log_accuracy_test = calculate_accuracy(output_test, class_labels_test);
            total_log_accuracy_test += log_accuracy_test;

        }

        if (total_log_accuracy == old_total_log_accuracy){
            alpha *= 2;
        }

        if (total_log_accuracy/10.0 >= 90){
            std::cout << "90% accuracy acheived on training data saving weights and bias to desktop, ending here :)" << std::endl;
            
            //save weights
            std::string weightz = create_weights_txtfile(weights,bias);
            std::cout << weightz;
            return 0;
            }

    }    





    return 0;
}
