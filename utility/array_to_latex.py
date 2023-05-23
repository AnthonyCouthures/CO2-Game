def array_to_latex(coefficients):
    # Initialize an empty string to build the LaTeX formula
    latex = ""

    # Loop through the coefficients in reverse order
    for i in range(len(coefficients)-1, -1, -1):
        # Check if the coefficient is zero; if so, skip to the next one
        if coefficients[i] == 0:
            continue
        
        # If the coefficient is positive and it's not the first term, add a plus sign
        if coefficients[i] > 0 and i < len(coefficients)-1:
            latex += "+"
        
        # If the coefficient is negative, add a minus sign
        if coefficients[i] < 0:
            latex += "-"
            # If the coefficient is -1 or 1, don't include it in the LaTeX output
            if abs(coefficients[i]) == 1:
                coefficients[i] = ""
        
        # If the coefficient is not 1 or -1, include it in the LaTeX output
        if abs(coefficients[i]) != 1:
            latex += str(abs(coefficients[i]))
        
        # If this is not the constant term, include the variable and its exponent
        if i > 0:
            latex += "x"
            if i > 1:
                latex += "^{" + str(i) + "}"
    
    # If the polynomial is identically zero, return "0"
    if latex == "":
        return "0"
    else:
        return latex