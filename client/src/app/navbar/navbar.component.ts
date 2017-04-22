import { Component, Input } from '@angular/core';
import { LoginComponent } from '../login/login.component.js';
import { AuthenticationService } from '../services/authentication.service.js'

@Component({
    selector: 'ss-navbar',
    templateUrl: 'src/app/navbar/navbar.component.html'
})

export class NavbarComponent {
    hideLoginBtn: boolean = false;

    constructor(private _authenticationService: AuthenticationService) { 
        _authenticationService.isUserLoggedIn$.subscribe(
            isUserLoggedIn => { this.hideLoginBtn = isUserLoggedIn }
        );
    }

   
}